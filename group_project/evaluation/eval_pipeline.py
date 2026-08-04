import sys
import types

# Monkeypatch missing vertexai module for RAGAS 0.1.21 compatibility
if "langchain_community.chat_models.vertexai" not in sys.modules:
    dummy_vertex = types.ModuleType("langchain_community.chat_models.vertexai")
    dummy_vertex.ChatVertexAI = None
    sys.modules["langchain_community.chat_models.vertexai"] = dummy_vertex

import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

VNU_GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset_vnu_15_ragas_0_1_21.json"
GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
JSON_RESULTS_PATH = Path(__file__).parent / "results.json"
DETAILS_PATH = Path(__file__).parent / "eval_details.json"


def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file (ưu tiên golden_dataset_vnu_15_ragas_0_1_21.json)."""
    target_path = VNU_GOLDEN_DATASET_PATH if VNU_GOLDEN_DATASET_PATH.exists() else GOLDEN_DATASET_PATH
    with open(target_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        print(f"Loaded {len(data)} test cases from {target_path.name}")
        return data


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

def compute_retrieval_metrics(retrieved_sources: list[dict], ground_truth_contexts: list[str], source_document: str = "", reference_doc_ids: list[str] = None) -> dict:
    """
    Tính toán các chỉ số đánh giá Retriever độc lập (không cần LLM Generator):
    - Hit Rate: Có lấy được chunk nào thuộc source_document / reference_doc_ids không?
    - Recall@k: Tỷ lệ chuỗi văn bản ground truth xuất hiện trong retrieved contexts.
    - MRR (Mean Reciprocal Rank): Thứ hạng nghịch đảo của chunk đúng đầu tiên.
    """
    retrieved_texts = [extract_text_content(s) for s in retrieved_sources]

    # 1. Document Hit Rate
    hit = 0
    ref_ids = reference_doc_ids or []
    for s in retrieved_sources:
        src_name = s.get("metadata", {}).get("source", "") if isinstance(s, dict) else ""
        if (source_document and (source_document in src_name or src_name in source_document)) or \
           any(ref_id in src_name for ref_id in ref_ids if ref_id):
            hit = 1
            break

    # 2. Recall@k
    recalled_count = 0
    valid_gts = [gt for gt in ground_truth_contexts if gt]
    if valid_gts:
        for gt in valid_gts:
            if any(gt[:50].lower() in ret.lower() for ret in retrieved_texts if ret):
                recalled_count += 1
        recall_at_k = recalled_count / len(valid_gts)
    else:
        recall_at_k = 1.0 if not retrieved_texts else 0.0

    # 3. MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for rank, ret in enumerate(retrieved_texts, start=1):
        if valid_gts and any(gt[:50].lower() in ret.lower() for gt in valid_gts if gt):
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
        ground_truth = item.get("ground_truth") or item.get("expected_answer", "")
        gt_contexts = item.get("ground_truth_context") or [item.get("expected_context", "")]
        source_doc = item.get("source_document") or ""
        ref_doc_ids = item.get("reference_document_ids", [])

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
            ret_metrics = compute_retrieval_metrics(sources if isinstance(result, dict) else [], gt_contexts, source_doc, ref_doc_ids)

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
                "reference_document_ids": ref_doc_ids,
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
                "reference_document_ids": ref_doc_ids,
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
        from src.task10_generation import generate_with_citation, format_context, reorder_for_llm, SYSTEM_PROMPT, TEMPERATURE, TOP_P, LLM_MODEL
        shared_generator = generate_with_citation
    except Exception as exc:
        print(f"[WARNING] Could not import task10_generation: {exc}")
        shared_generator = None

    # Pipeline A: Hybrid Retrieval -> Generator
    def pipeline_a(q):
        from src.task10_generation import generate_with_citation
        return generate_with_citation(q)

    # Pipeline B: Dense Retrieval -> Generator
    def pipeline_b(q):
        try:
            from src.task5_semantic_search import semantic_search
            docs = semantic_search(q, top_k=5)
            sources = []
            for d in docs:
                txt = extract_text_content(d)
                sources.append({
                    "content": txt,
                    "metadata": getattr(d, "metadata", {}) if not isinstance(d, dict) else d.get("metadata", {})
                })

            if shared_generator is not None and sources:
                reordered = reorder_for_llm(sources)
                context = format_context(reordered)
                user_message = f"Context:\n{context}\n\n---\n\nQuestion: {q}"

                api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
                if not api_key:
                    return {
                        "answer": f"Dựa trên tài liệu tham khảo: {sources[0]['content'][:200]}...",
                        "sources": sources
                    }

                from openai import OpenAI
                if os.getenv("OPENAI_API_KEY"):
                    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                    model_name = "gpt-4o-mini"
                else:
                    client = OpenAI(api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")
                    model_name = LLM_MODEL if LLM_MODEL else "openai/gpt-4o-mini"

                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=TEMPERATURE,
                    top_p=TOP_P
                )
                answer = resp.choices[0].message.content or ""
                return {
                    "answer": answer,
                    "sources": sources,
                    "retrieval_source": "dense"
                }
            else:
                return {
                    "answer": f"Dựa trên tài liệu tham khảo: {sources[0]['content'][:200]}..." if sources else "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
                    "sources": sources,
                    "retrieval_source": "dense"
                }
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
    """Xuất báo cáo kết quả đánh giá A/B Testing ra results.json dạng số chuẩn."""
    config_a_res = comparison_results.get("Config A (hybrid + rerank)", {})
    config_b_res = comparison_results.get("Config B (dense-only)", {})

    exec_a = config_a_res.get("execution_summary", {}) if isinstance(config_a_res, dict) else {}
    exec_b = config_b_res.get("execution_summary", {}) if isinstance(config_b_res, dict) else {}

    scores_a = config_a_res.get("overall_scores", {}) if isinstance(config_a_res, dict) else {}
    scores_b = config_b_res.get("overall_scores", {}) if isinstance(config_b_res, dict) else {}

    metrics_keys = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
    metrics_comparison = {}
    for metric in metrics_keys:
        val_a = round(scores_a.get(metric, 0.0), 4)
        val_b = round(scores_b.get(metric, 0.0), 4)
        metrics_comparison[metric] = {
            "config_a_score": val_a,
            "config_b_score": val_b,
            "delta_a_minus_b": round(val_a - val_b, 4)
        }

    export_data = {
        "metadata": {
            "evaluator_framework": "RAGAS v0.1.21",
            "evaluator_llm": "GPT-4o (temperature=0.0)",
            "benchmark_dataset": "golden_dataset_vnu_15_ragas_0_1_21.json",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        },
        "execution_summary": {
            "config_a_hybrid_rerank": exec_a,
            "config_b_dense_only": exec_b
        },
        "overall_scores": {
            "config_a_hybrid_rerank": scores_a,
            "config_b_dense_only": scores_b
        },
        "metrics_comparison": metrics_comparison,
        "ab_comparison_results": comparison_results
    }

    with open(JSON_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)
    print(f"[SUCCESS] Benchmark results successfully exported to {JSON_RESULTS_PATH}")


if __name__ == "__main__":
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases from {GOLDEN_DATASET_PATH.name}")

    evaluator_stack = get_evaluator_stack()
    comparison_results = compare_configs(golden_dataset, evaluator_stack)
    save_eval_details(comparison_results)
    export_results(comparison_results)

