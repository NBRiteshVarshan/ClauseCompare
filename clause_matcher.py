import time
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ollama
from utils import generate_clause_id

class LegalClauseMatcher:
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        print(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        self.cache = {}
        self.llm_model = 'qwen2.5:7b'

    def get_embeddings(self, clauses: List[Dict], doc_name: str) -> np.ndarray:
        embeddings = []
        for clause in clauses:
            cid = generate_clause_id(clause['text'], doc_name)
            if cid in self.cache:
                embeddings.append(self.cache[cid])
            else:
                emb = self.embedder.encode(clause['text'])
                self.cache[cid] = emb
                embeddings.append(emb)
        return np.array(embeddings)

    def compare_with_llm(self, clause_a, clause_b):
        try:
            prompt = """You are comparing two legal clauses. They match if:
1. They create the same obligation (who must do what)
2. They grant the same right or power
3. They have the same conditions and exceptions
4. The consequences of violation are the same

They DO NOT match if:
- One has additional conditions the other lacks
- The scope is different (e.g., "worldwide" vs "US only")
- The obligation is stronger/weaker (e.g., "shall" vs "may")

Clause A: "{clause_a}"
Clause B: "{clause_b}"

Respond in JSON format with these keys:
- "match": true/false (boolean)
- "confidence": 0-1 (float)
- "key_differences": ["difference1", "difference2"] (list)
- "reason": "brief explanation" (string)"""
            response = ollama.chat(
                model=self.llm_model,
                messages=[
                    {'role': 'system', 'content': 'You are a legal text comparison expert. Respond only in JSON format.'},
                    {'role': 'user', 'content': prompt.format(clause_a=clause_a[:500], clause_b=clause_b[:500])}
                ]
            )
            return json.loads(response['message']['content'])
        except Exception as e:
            print(f"LLM error: {e}")
            return {'match': False, 'confidence': 0.5, 'key_differences': [], 'reason': 'LLM failed'}

    def match_documents(self, doc1_clauses, doc2_clauses,
                        doc1_name="Document 1", doc2_name="Document 2",
                        similarity_threshold=0.4,
                        high_similarity_threshold=0.85,
                        match_threshold=0.5):
        
        start = time.time()
        print("Generating embeddings...")
        emb1 = self.get_embeddings(doc1_clauses, doc1_name)
        emb2 = self.get_embeddings(doc2_clauses, doc2_name)

        n1 = len(doc1_clauses)
        n2 = len(doc2_clauses)

        # Full similarity matrix
        sim_matrix = cosine_similarity(emb1, emb2)

        # Track matches
        matched_doc1 = [False] * n1
        matched_doc2 = [False] * n2
        
        # Store match details for each doc1 clause
        match_pairs = []  # list of (doc1_idx, doc2_idx, similarity, reason)

        # STEP 1: Mutual best matching
        print("Step 1: Finding mutual best matches...")
        best1 = np.argmax(sim_matrix, axis=1)
        best_sim1 = sim_matrix[np.arange(n1), best1]
        best2 = np.argmax(sim_matrix, axis=0)

        mutual_count = 0
        for i in range(n1):
            j = best1[i]
            sim = best_sim1[i]
            if best2[j] == i and sim >= match_threshold:
                if not matched_doc1[i] and not matched_doc2[j]:
                    matched_doc1[i] = True
                    matched_doc2[j] = True
                    match_pairs.append((i, j, sim, 'Mutual best match'))
                    mutual_count += 1

        print(f"  → {mutual_count} mutual best matches found")

        # STEP 2: Greedy matching (Doc1 → Doc2)
        print("Step 2: Greedy matching remaining clauses...")
        greedy_count = 0
        llm_count = 0
        
        for i in range(n1):
            if matched_doc1[i]:
                continue  # already matched

            # Get top candidates for this doc1 clause
            sims = sim_matrix[i, :]  # similarities to all doc2 clauses
            # Sort by similarity descending
            sorted_indices = np.argsort(sims)[::-1]
            
            matched = False
            best_match_info = None

            for j in sorted_indices:
                if matched_doc2[j]:
                    continue  # this doc2 clause is already taken
                
                sim = sims[j]
                
                if sim < similarity_threshold:
                    break  # no more candidates above threshold

                if sim >= high_similarity_threshold:
                    # High similarity – instant match
                    matched_doc1[i] = True
                    matched_doc2[j] = True
                    match_pairs.append((i, j, sim, f'High similarity (≥{high_similarity_threshold})'))
                    matched = True
                    greedy_count += 1
                    break

                # Borderline – use LLM
                clause1 = doc1_clauses[i]
                clause2 = doc2_clauses[j]
                llm_result = self.compare_with_llm(clause1['text'], clause2['text'])
                
                if llm_result.get('match', False) and llm_result.get('confidence', 0) > 0.7:
                    matched_doc1[i] = True
                    matched_doc2[j] = True
                    match_pairs.append((i, j, sim, llm_result.get('reason', 'LLM match')))
                    matched = True
                    llm_count += 1
                    break

            if not matched:
                # This doc1 clause remains unmatched
                pass

        print(f"  → {greedy_count} greedy matches, {llm_count} via LLM")

        # Build the results
        matching_details = []
        only_in_doc1 = []
        
        # Map doc1 index to its match (doc2 index and similarity)
        match_map = {}
        for i, j, sim, reason in match_pairs:
            match_map[i] = (j, sim, reason)

        for i, clause1 in enumerate(doc1_clauses):
            if i in match_map:
                j, sim, reason = match_map[i]
                clause2 = doc2_clauses[j]
                matching_details.append({
                    'clause_number': clause1.get('number', str(i+1)),
                    'clause_text': clause1['text'],
                    'found_match': True,
                    'best_match': {
                        'clause_idx': j,
                        'similarity': sim,
                        'confidence': 1.0 if 'high' in reason.lower() or 'mutual' in reason.lower() else 0.9,
                        'reason': reason,
                        'key_differences': [],
                        'used_llm': 'LLM' in reason
                    },
                    'top_similarity': sim,
                    'top_match_idx': j
                })
            else:
                # Unmatched doc1
                top_j = np.argmax(sim_matrix[i, :])
                top_sim = sim_matrix[i, top_j]
                matching_details.append({
                    'clause_number': clause1.get('number', str(i+1)),
                    'clause_text': clause1['text'],
                    'found_match': False,
                    'best_match': None,
                    'top_similarity': top_sim,
                    'top_match_idx': top_j
                })
                only_in_doc1.append({
                    'text': clause1['text'],
                    'number': clause1.get('number', str(i+1)),
                    'closest_match': doc2_clauses[top_j]['text'] if top_j < n2 else "",
                    'similarity': top_sim,
                    'metadata': clause1.get('metadata', {})
                })

        # Collect unmatched doc2 clauses
        only_in_doc2 = []
        for j in range(n2):
            if not matched_doc2[j]:
                sims = cosine_similarity([emb2[j]], emb1)[0]
                best_idx = np.argmax(sims)
                only_in_doc2.append({
                    'text': doc2_clauses[j]['text'],
                    'number': doc2_clauses[j].get('number', str(j+1)),
                    'closest_match': doc1_clauses[best_idx]['text'] if best_idx < n1 else "",
                    'similarity': sims[best_idx],
                    'metadata': doc2_clauses[j].get('metadata', {})
                })

        elapsed = time.time() - start
        matching_count = len(match_pairs)
        print(f"✅ Final: {matching_count} matches total")

        return {
            'only_in_doc1': only_in_doc1,
            'only_in_doc2': only_in_doc2,
            'matching_details': matching_details,
            'total_doc1': n1,
            'total_doc2': n2,
            'matching_count': matching_count,
            'processing_time': elapsed,
            'similarity_threshold': similarity_threshold,
            'high_similarity_threshold': high_similarity_threshold,
            'llm_matches': llm_count,
            'high_sim_matches': greedy_count + mutual_count,
            'doc2_best_similarities': [0.0] * n2
        }