from typing import Dict, List, Tuple
from ..llm.client import chat_structured, embed
from ..llm.schemas import TextRelevance
import numpy as np

def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

def llm_judge_merge(main_topic_label: str, subtopic_label: str) -> Tuple[bool, str]:
    """
    Uses an LLM to judge if a subtopic should be merged with a main topic and suggest a merged label.
    Returns (should_merge, suggested_label).
    """
    _MERGE_SYS = (
        "You are a research analyst. Your task is to determine if a specific research concept should be "
        "merged with a broader research topic, and if so, suggest an improved label for the merged topic."
    )
    
    _MERGE_USER_TPL = (
        "MAIN TOPIC: {main_topic}\n"
        "SUBTopic: {subtopic}\n\n"
        "Should '{subtopic}' be merged into '{main_topic}' as a sub-concept? "
        "If yes, suggest a better label for the merged topic that captures both concepts. "
        "If no, respond with 'NO MERGE'."
    )
    
    prompt = _MERGE_USER_TPL.format(main_topic=main_topic_label, subtopic=subtopic_label)
    
    try:
        response = chat_structured(
            response_model=TextRelevance,
            system=_MERGE_SYS,
            user=prompt,
            temperature=0.1
        )
        
        # Parse the response to determine if merge should happen
        # is_relevant is a boolean, so we use it directly
        should_merge = response.is_relevant if hasattr(response, 'is_relevant') else False
        suggested_label = main_topic_label  # Default to original if no better suggestion
        
        # Use both the boolean decision and confidence score
        return should_merge and response.confidence > 0.6, suggested_label
        
    except Exception as e:
        print(f"Error in LLM merge judgment: {e}")
        return False, main_topic_label

def calculate_dynamic_threshold(similarities: List[float]) -> float:
    """
    Calculates a dynamic threshold based on the distribution of similarity scores.
    """
    if len(similarities) < 2:
        return 0.6  # Default threshold
    
    similarities = np.array(similarities)
    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    
    # Dynamic threshold: mean + 1.5 * standard deviation
    # This ensures only statistically significant similarities are considered
    dynamic_threshold = mean_sim + 1.5 * std_sim
    
    # Clamp between reasonable bounds
    return max(0.4, min(0.8, dynamic_threshold))

def merge_subtopics_advanced(per_paper_results: List[Dict], corpus_topics: List[Dict], max_passes: int = 3) -> Dict[str, str]:
    """
    Advanced multi-pass merging with LLM judgment and dynamic thresholds.
    """
    clustered_paper_topic_ids = {member_id for t in corpus_topics for member_id in t.get('member_topic_ids', [])}
    
    all_paper_topics = []
    for paper in per_paper_results:
        for topic in paper.get("topics", []):
            all_paper_topics.append({"id": topic["id"], "label": topic["label"]})
            
    unmatched_topics = [t for t in all_paper_topics if t['id'] not in clustered_paper_topic_ids]
    
    if not unmatched_topics or not corpus_topics:
        return {}
    
    subtopic_mapping = {}
    current_corpus_topics = corpus_topics.copy()
    
    for pass_num in range(max_passes):
        print(f"\nMerge Pass {pass_num + 1}: Processing {len(unmatched_topics)} unmatched topics...")
        
        if not unmatched_topics:
            break
            
        # Get embeddings for current iteration
        main_topic_labels = [t['label'] for t in current_corpus_topics]
        unmatched_topic_labels = [t['label'] for t in unmatched_topics]
        
        main_topic_vecs = embed(main_topic_labels)
        unmatched_topic_vecs = embed(unmatched_topic_labels)
        
        # Calculate all similarities for dynamic threshold
        all_similarities = []
        for i, un_topic in enumerate(unmatched_topics):
            un_vec = unmatched_topic_vecs[i]
            for j, main_topic in enumerate(current_corpus_topics):
                main_vec = main_topic_vecs[j]
                sim = cos_sim(un_vec, main_vec)
                all_similarities.append(sim)
        
        # Calculate dynamic threshold
        dynamic_threshold = calculate_dynamic_threshold(all_similarities)
        print(f"  Dynamic threshold for pass {pass_num + 1}: {dynamic_threshold:.3f}")
        
        # Process each unmatched topic
        topics_to_remove = []
        for i, un_topic in enumerate(unmatched_topics):
            un_vec = unmatched_topic_vecs[i]
            best_sim = -1
            best_match_idx = None
            
            # Find best match
            for j, main_topic in enumerate(current_corpus_topics):
                main_vec = main_topic_vecs[j]
                sim = cos_sim(un_vec, main_vec)
                
                if sim > best_sim:
                    best_sim = sim
                    best_match_idx = j
            
            # Check if similarity meets dynamic threshold
            if best_sim >= dynamic_threshold:
                main_topic = current_corpus_topics[best_match_idx]
                
                # Use LLM to judge the merge
                should_merge, suggested_label = llm_judge_merge(main_topic['label'], un_topic['label'])
                
                if should_merge:
                    print(f"  ✓ Merging '{un_topic['label']}' into '{main_topic['label']}' (sim: {best_sim:.3f})")
                    subtopic_mapping[un_topic['id']] = main_topic['id']
                    topics_to_remove.append(i)
                else:
                    print(f"  ✗ LLM rejected merge of '{un_topic['label']}' into '{main_topic['label']}' (sim: {best_sim:.3f})")
        
        # Remove successfully merged topics
        for idx in reversed(topics_to_remove):
            unmatched_topics.pop(idx)
        
        print(f"  Pass {pass_num + 1} complete: {len(topics_to_remove)} topics merged, {len(unmatched_topics)} remaining")
    
    return subtopic_mapping
