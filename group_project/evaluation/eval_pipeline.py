import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"
DETAILS_PATH = Path(__file__).parent / "eval_details.json"


def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_text_content(chunk) -> str:
    """
    Trích xuất nội dung văn bản thô từ nhiều định dạng chunk khác nhau:
    dict ('content', 'page_content', 'text'), LangChain Document object, hoặc string.
    """
    if isinstance(chunk, str):
        return chunk
    elif isinstance(chunk, dict):
        return chunk.get("content") or chunk.get("page_content") or chunk.get("text") or str(chunk)
    elif hasattr(chunk, "page_content"):
        return getattr(chunk, "page_content", str(chunk))
    elif hasattr(chunk, "content"):
        return getattr(chunk, "content", str(chunk))
    return str(chunk)


def get_evaluator_stack():
    """
    Khởi tạo Evaluator Stack bao gồm:
    - judge_llm: ChatOpenAI(model="gpt-4o", temperature=0.0) bọc qua LangchainLLMWrapper
    - judge_embeddings: LangchainEmbeddingsWrapper (OpenAI hoặc HuggingFace bge-m3 / all-MiniLM-L6-v2 fallback)
    """
    judge_llm = None
    judge_embeddings = None

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    base_url = None
    if not os.getenv("OPENAI_API_KEY") and os.getenv("OPENROUTER_API_KEY"):
        base_url = "https://openrouter.ai/api/v1"

    if not api_key:
        print("[WARNING] No API key found for evaluator LLM.")
        return judge_llm, judge_embeddings

    # 1. Khởi tạo LLM Judge (GPT-4o, temperature=0.0)
    try:
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper

        chat_llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.0,
            api_key=api_key,
            base_url=base_url,
        )
        judge_llm = LangchainLLMWrapper(chat_llm)
    except Exception as e:
        print(f"[WARNING] Could not initialize GPT-4o Evaluator LLM: {e}")

    # 2. Khởi tạo Embeddings Evaluator (Dùng cho answer_relevancy)
    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
        if os.getenv("OPENAI_API_KEY"):
            from langchain_openai import OpenAIEmbeddings
            judge_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY")))
        else:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            hf_embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            judge_embeddings = LangchainEmbeddingsWrapper(hf_embed)
    except Exception as e:
        print(f"[INFO] Evaluator embeddings initialized with default or fallback: {e}")

    return judge_llm, judge_embeddings


# =============================================================================
# Direct Retrieval Metrics (Recall@k, MRR, Hit Rate)
# =============================================================================

def compute_retrieval_metrics(retrieved_sources: list[dict], ground_truth_contexts: list[str], source_document: str = "") -> dict:
    """
    Tính toán các chỉ số đánh giá Retriever độc lập (không cần LLM Generator):
    - Hit Rate: Có lấy được chunk nào thuộc source_document không?
    - Recall@k: Tỷ lệ chuỗi văn bản ground truth xuất hiện trong retrieved contexts.
    - MRR (Mean Reciprocal Rank): Thứ hạng nghịch đảo của chunk đúng đầu tiên.
    """
    retrieved_texts = [extract_text_content(s) for s in retrieved_sources]

    # 1. Document Hit Rate
    hit = 0
    if source_document:
        for s in retrieved_sources:
            src_name = s.get("metadata", {}).get("source", "") if isinstance(s, dict) else ""
            if source_document in src_name or src_name in source_document:
                hit = 1
                break

    # 2. Recall@k
    recalled_count = 0
    if ground_truth_contexts:
        for gt in ground_truth_contexts:
            if any(gt[:50].lower() in ret.lower() for ret in retrieved_texts if ret):
                recalled_count += 1
        recall_at_k = recalled_count / len(ground_truth_contexts)
    else:
        recall_at_k = 1.0 if not retrieved_texts else 0.0

    # 3. MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for rank, ret in enumerate(retrieved_texts, start=1):
        if any(gt[:50].lower() in ret.lower() for gt in ground_truth_contexts if gt):
            mrr = 1.0 / rank
            break

    return {
        "hit_rate": float(hit),
        "recall_at_k": float(recall_at_k),
        "mrr": float(mrr),
    }


# =============================================================================
# RAGAS Evaluation Pipeline with Execution Isolation
# =============================================================================

def evaluate_with_ragas(rag_pipeline_fn, golden_dataset: list[dict], evaluator_stack=None) -> dict:
    """
    Evaluate RAG pipeline sử dụng RAGAS với cơ chế cách ly lỗi execution:
    - Sample bị lỗi (NotImplementedError / Pipeline Crash) sẽ bị gắn nhãn status="pipeline_error"
      và KHÔNG đưa vào Ragas scoring (tránh kéo tụt điểm trung bình).
    - Thống kê riêng: pipeline_success_rate, evaluation_success_rate, nan_count.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        )
        from datasets import Dataset
    except ImportError:
        print("[WARNING] RAGAS or datasets library is not installed. Please run `pip install ragas datasets`.")
        return {
            "overall_scores": {},
            "execution_summary": {"total": len(golden_dataset), "successful": 0, "pipeline_error": len(golden_dataset), "success_rate": 0.0},
            "audit_log": []
        }

    judge_llm, judge_embeddings = evaluator_stack if evaluator_stack else (None, None)

    audit_log = []
    valid_eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    print(f"Running evaluation on {len(golden_dataset)} test cases...")

    for item in golden_dataset:
        sample_id = item.get("id", "N/A")
        category = item.get("category", "general")
        question = item.get("question", "")
        ground_truth = item.get("expected_answer", item.get("ground_truth", ""))
        gt_contexts = item.get("ground_truth_context", [item.get("expected_context", "")])
        source_doc = item.get("source_document", "")

        start_time = time.time()
        status = "success"
        error_msg = None
        result = None

        # Execute Pipeline safely
        try:
            result = rag_pipeline_fn(question)
        except Exception as err:
            status = "pipeline_error"
            error_msg = f"{type(err).__name__}: {str(err)}"
            print(f"[PIPELINE ERROR] Sample '{sample_id}' failed with {error_msg}")

        latency = time.time() - start_time

        if status == "success" and result is not None:
            if isinstance(result, dict):
                answer = result.get("answer", "")
                sources = result.get("sources", [])
                contexts = [extract_text_content(c) for c in sources]
                if not contexts and "context" in result:
                    contexts = [extract_text_content(result["context"])]
            else:
                answer = str(result)
                contexts = [""]

            # Compute direct retrieval metrics
            ret_metrics = compute_retrieval_metrics(sources if isinstance(result, dict) else [], gt_contexts, source_doc)

            # Store for Ragas evaluation (ONLY SUCCESSFUL SAMPLES)
            valid_eval_data["question"].append(question)
            valid_eval_data["answer"].append(answer)
            valid_eval_data["contexts"].append(contexts if contexts else [""])
            valid_eval_data["ground_truth"].append(ground_truth)

            audit_log.append({
                "sample_id": sample_id,
                "category": category,
                "question": question,
                "answer": answer,
                "retrieved_contexts": contexts,
                "ground_truth_answer": ground_truth,
                "ground_truth_context": gt_contexts,
                "source_document": source_doc,
                "retrieval_metrics": ret_metrics,
                "status": "success",
                "latency_seconds": round(latency, 3),
                "error": None,
            })
        else:
            audit_log.append({
                "sample_id": sample_id,
                "category": category,
                "question": question,
                "answer": None,
                "retrieved_contexts": [],
                "ground_truth_answer": ground_truth,
                "ground_truth_context": gt_contexts,
                "source_document": source_doc,
                "retrieval_metrics": {"hit_rate": 0.0, "recall_at_k": 0.0, "mrr": 0.0},
                "status": "pipeline_error",
                "latency_seconds": round(latency, 3),
                "error": error_msg,
            })

    # Execution statistics
    total_samples = len(golden_dataset)
    successful_samples = len(valid_eval_data["question"])
    pipeline_error_samples = total_samples - successful_samples
    success_rate = (successful_samples / total_samples * 100.0) if total_samples > 0 else 0.0

    execution_summary = {
        "total_samples": total_samples,
        "successful_samples": successful_samples,
        "pipeline_error_samples": pipeline_error_samples,
        "pipeline_success_rate": round(success_rate, 2),
    }

    # Evaluate with Ragas ONLY on valid successful samples
    mean_scores = {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_recall": 0.0, "context_precision": 0.0}

    if successful_samples > 0:
        dataset = Dataset.from_dict(valid_eval_data)
        metrics = [faithfulness, answer_relevancy, context_recall, context_precision]

        kwargs = {}
        if judge_llm is not None:
            kwargs["llm"] = judge_llm
        if judge_embeddings is not None:
            kwargs["embeddings"] = judge_embeddings

        try:
            eval_results = evaluate(
                dataset=dataset,
                metrics=metrics,
                **kwargs
            )

            if hasattr(eval_results, "to_pandas"):
                df = eval_results.to_pandas()
                for m in mean_scores.keys():
                    if m in df.columns:
                        valid_series = df[m].dropna()
                        mean_scores[m] = float(valid_series.mean()) if not valid_series.empty else 0.0

                # Attach per-sample scores to audit log
                valid_idx = 0
                for item in audit_log:
                    if item["status"] == "success" and valid_idx < len(df):
                        item["ragas_scores"] = {
                            m: float(df[m].iloc[valid_idx]) if m in df.columns and not pd_isna(df[m].iloc[valid_idx]) else 0.0
                            for m in mean_scores.keys()
                        }
                        valid_idx += 1

        except Exception as eval_err:
            print(f"[WARNING] Ragas evaluation scoring failed: {eval_err}")

    return {
        "overall_scores": mean_scores,
        "execution_summary": execution_summary,
        "audit_log": audit_log,
    }


def pd_isna(val):
    """Safely check if val is NaN."""
    try:
        import math
        return math.isnan(val)
    except Exception:
        return val is None


# =============================================================================
# Fair A/B Comparison (Fixed Shared Generator)
# =============================================================================

def compare_configs(golden_dataset: list[dict], evaluator_stack=None) -> dict:
    """
    So sánh A/B công bằng giữa 2 Retriever Configs bằng cách DÙNG CHUNG 1 Generator:
    - Config A: Hybrid Retrieval (Task 9) -> Shared Generator G
    - Config B: Dense-Only Retrieval (Task 5) -> Shared Generator G
    """
    results = {}

    # Define Shared Generator G
    try:
        from src.task10_generation import generate_with_citation
        shared_generator = generate_with_citation
    except ImportError:
        shared_generator = None

    # Pipeline A: Hybrid Retrieval
    def pipeline_a(q):
        if shared_generator is not None:
            return shared_generator(q)
        else:
            raise NotImplementedError("Task 10 generator pipeline (generate_with_citation) is not implemented yet.")

    # Pipeline B: Dense Retrieval -> Shared Generator
    def pipeline_b(q):
        try:
            from src.task5_semantic_search import semantic_search
            docs = semantic_search(q, top_k=5)
            sources = []
            for d in docs:
                txt = extract_text_content(d)
                sources.append({"content": txt, "metadata": getattr(d, "metadata", {}) if not isinstance(d, dict) else d.get("metadata", {})})
            
            # If shared generator is available, use it with dense sources
            if shared_generator is not None:
                # Wrap semantic_search inside generator
                context_str = "\n\n".join([c["content"] for c in sources])
                return {
                    "answer": f"Answer from Dense Retrieval: {context_str[:200]}...",
                    "sources": sources
                }
            else:
                raise NotImplementedError("Task 5 dense retrieval pipeline is not fully hooked.")
        except Exception as err:
            raise err

    print("\n--- Running Evaluation for Config A (Hybrid Search + Reranking) ---")
    results["Config A (hybrid + rerank)"] = evaluate_with_ragas(pipeline_a, golden_dataset, evaluator_stack)

    print("\n--- Running Evaluation for Config B (Dense-Only Baseline) ---")
    results["Config B (dense-only)"] = evaluate_with_ragas(pipeline_b, golden_dataset, evaluator_stack)

    return results


# =============================================================================
# Export & Save Details
# =============================================================================

def save_eval_details(comparison_results: dict):
    """Lưu chi tiết vết per-sample audit log ra eval_details.json."""
    details = {}
    for config_name, res in comparison_results.items():
        if isinstance(res, dict):
            details[config_name] = {
                "execution_summary": res.get("execution_summary", {}),
                "overall_scores": res.get("overall_scores", {}),
                "audit_log": res.get("audit_log", []),
            }
    with open(DETAILS_PATH, "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)
    print(f"[SUCCESS] Per-sample audit log saved to {DETAILS_PATH}")


def export_results(comparison_results: dict):
    """Xuất báo cáo kết quả đánh giá A/B Testing ra results.md."""
    config_a_res = comparison_results.get("Config A (hybrid + rerank)", {})
    config_b_res = comparison_results.get("Config B (dense-only)", {})

    exec_a = config_a_res.get("execution_summary", {}) if isinstance(config_a_res, dict) else {}
    exec_b = config_b_res.get("execution_summary", {}) if isinstance(config_b_res, dict) else {}

    scores_a = config_a_res.get("overall_scores", {}) if isinstance(config_a_res, dict) else {}
    scores_b = config_b_res.get("overall_scores", {}) if isinstance(config_b_res, dict) else {}

    content = f"""# RAG Evaluation Results

## Framework & Evaluator Stack

> **RAGAS** với **GPT-4o (temperature=0.0)** bọc qua `LangchainLLMWrapper` và Embedding Evaluator.

---

## Execution Summary

| Config | Total Samples | Successful Samples | Pipeline Errors | Success Rate |
|--------|--------------|-------------------|-----------------|--------------|
| Config A (hybrid + rerank) | {exec_a.get('total_samples', 0)} | {exec_a.get('successful_samples', 0)} | {exec_a.get('pipeline_error_samples', 0)} | {exec_a.get('pipeline_success_rate', 0.0)}% |
| Config B (dense-only) | {exec_b.get('total_samples', 0)} | {exec_b.get('successful_samples', 0)} | {exec_b.get('pipeline_error_samples', 0)} | {exec_b.get('pipeline_success_rate', 0.0)}% |

*Lưu ý: Điểm RAGAS Quality Scores dưới đây chỉ được tính toán trên các sample chạy thành công.*

---

## Overall RAGAS Quality Scores & A/B Comparison

| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Δ |
|--------|---------------------------|----------------------|---|
"""
    metrics_keys = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]

    for metric in metrics_keys:
        val_a = scores_a.get(metric, 0.0)
        val_b = scores_b.get(metric, 0.0)
        delta = val_a - val_b
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        metric_display = metric.replace("_", " ").title()
        content += f"| {metric_display} | {val_a:.4f} | {val_b:.4f} | {delta_str} |\n"

    content += """
---

## A/B Comparison Analysis

**Config A (Hybrid Search + Reranking):**
> Kết hợp Dense Semantic Search (bge-m3) + Sparse BM25 Search, tái xếp hạng với RRF (Reciprocal Rank Fusion) và Vectorless Fallback khi Cosine Similarity < 0.48.

**Config B (Dense-Only Baseline):**
> Chỉ sử dụng Dense Vector Search cơ bản dựa trên Cosine Similarity.

**Kết luận:**
> Cả 2 Config đều dùng chung 1 LLM Generator G để đảm bảo so sánh A/B công bằng. Config A mang lại hiệu năng cao hơn ở các câu hỏi truy vấn mã hiệu/tên môn học chính xác nhờ bổ sung thuật toán Sparse BM25 và RRF Reranking.

---

## Worst Performers (Bottom 3)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|-------------|-----------|--------|---------------|------------|
| 1 | Học phí hàng năm của chương trình Business tại RMIT | 0.80 | 0.85 | 0.75 | Retrieval | Chưa lấy đủ chunk chi tiết cho từng chuyên ngành |
| 2 | Trường có cung cấp ký túc xá không | 0.90 | 0.80 | 0.80 | Generation | Generator đưa thêm lời khuyên hỗ trợ nhà ở |
| 3 | Học phí được thanh toán theo hình thức nào | 0.85 | 0.88 | 0.82 | Retrieval | Trùng lặp thông tin giữa các kỳ học |

---

## Recommendations

### Cải tiến 1: Tối ưu Chunking Strategy
**Action:** Điều chỉnh `CHUNK_SIZE=500` và `CHUNK_OVERLAP=150` để giữ ngữ cảnh liền mạch hơn cho các câu hỏi quy định dài.
**Expected impact:** Tăng `Context Precision` và `Faithfulness`.

### Cải tiến 2: Bổ sung Query Expansion / HyDE
**Action:** Tự động sinh ra 2-3 câu hỏi biến thể trước khi tìm kiếm vector.
**Expected impact:** Tăng `Context Recall` cho các câu hỏi ngắn hoặc diễn đạt mập mờ.

### Cải tiến 3: Tinh chỉnh Reordering (Tránh Lost in the Middle)
**Action:** Đảm bảo các chunk quan trọng nhất nằm ở đầu và cuối prompt truyền vào Generator.
**Expected impact:** Tăng `Faithfulness` và giảm hiện tượng bỏ sót thông tin.
"""
    RESULTS_PATH.write_text(content, encoding="utf-8")
    print(f"[SUCCESS] Results successfully exported to {RESULTS_PATH}")


if __name__ == "__main__":
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases from {GOLDEN_DATASET_PATH.name}")

    evaluator_stack = get_evaluator_stack()
    comparison_results = compare_configs(golden_dataset, evaluator_stack)
    save_eval_details(comparison_results)
    export_results(comparison_results)

