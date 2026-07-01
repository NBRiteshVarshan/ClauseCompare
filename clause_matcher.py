import time
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ollama
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import generate_clause_id

class LegalClauseMatcher:
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        print(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        self.cache = {}
        self.llm_model = 'llama3.2:3b'

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
        """Synchronous LLM call – used by ThreadPoolExecutor."""
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
            return {'match': False, 'confidence': 0.0, 'reason': str(e)}

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

        # ------------------------------------------------------------
        # STEP 1: Find best match for each Doc1 clause (≥ 0.5)
        # Build map: Doc2[j] → list of (Doc1_idx, similarity)
        # ------------------------------------------------------------
        best_match_map = {}  # key: Doc2 index, value: list of (Doc1_idx, similarity)

        for i in range(n1):
            # Find the best match for this Doc1 clause
            best_j = np.argmax(sim_matrix[i, :])
            best_sim = sim_matrix[i, best_j]

            if best_sim >= match_threshold:  # ≥ 0.5
                if best_j not in best_match_map:
                    best_match_map[best_j] = []
                best_match_map[best_j].append((i, best_sim))

        print(f"Step 1: {len(best_match_map)} Doc2 clauses claimed as best match by Doc1 clauses")

        # ------------------------------------------------------------
        # STEP 2: Resolve conflicts – pick highest similarity per Doc2
        # ------------------------------------------------------------
        matched_doc1 = [False] * n1
        matched_doc2 = [False] * n2
        match_pairs = []  # list of (doc1_idx, doc2_idx, similarity, reason)

        for j, claimants in best_match_map.items():
            if len(claimants) == 1:
                # Only one Doc1 clause claims this Doc2 – pair them
                i, sim = claimants[0]
                matched_doc1[i] = True
                matched_doc2[j] = True
                match_pairs.append((i, j, sim, 'Best match (unique claimant)'))
            else:
                # Multiple Doc1 clauses claim this Doc2 – pick the highest similarity
                # Sort by similarity descending
                claimants.sort(key=lambda x: x[1], reverse=True)
                winner_i, winner_sim = claimants[0]
                matched_doc1[winner_i] = True
                matched_doc2[j] = True
                match_pairs.append((winner_i, j, winner_sim, 'Best match (highest similarity among claimants)'))

        print(f"Step 2: {len(match_pairs)} direct matches assigned from best-match mapping")

        # ------------------------------------------------------------
        # STEP 3: Handle remaining unmatched Doc1 clauses (greedy fallback)
        # ------------------------------------------------------------
        ambiguous_pairs = []  # list of (doc1_idx, doc2_idx, similarity)

        for i in range(n1):
            if matched_doc1[i]:
                continue

            # Get top candidates for this doc1 clause
            sims = sim_matrix[i, :]
            sorted_indices = np.argsort(sims)[::-1]

            matched = False

            for j in sorted_indices:
                if matched_doc2[j]:
                    continue

                sim = sims[j]

                if sim < similarity_threshold:
                    break  # no more candidates above threshold

                if sim >= high_similarity_threshold:
                    # High similarity – instant match
                    matched_doc1[i] = True
                    matched_doc2[j] = True
                    match_pairs.append((i, j, sim, f'High similarity (≥{high_similarity_threshold})'))
                    matched = True
                    break

                elif sim >= similarity_threshold:
                    # Borderline – collect for LLM verification
                    ambiguous_pairs.append((i, j, sim))

            # If matched, we're done; otherwise it will go to LLM or remain unique

        print(f"Step 3: {len(ambiguous_pairs)} ambiguous candidates for LLM")

        # ------------------------------------------------------------
        # STEP 4: Process ambiguous pairs with LLM (concurrent)
        # ------------------------------------------------------------
        if ambiguous_pairs:
            print(f"⚡ Processing {len(ambiguous_pairs)} ambiguous candidates with LLM...")

            # Sort by similarity descending (higher chance of match first)
            ambiguous_pairs.sort(key=lambda x: x[2], reverse=True)

            # Filter: only include candidates where both clauses are still unmatched
            filtered_pairs = []
            for i, j, sim in ambiguous_pairs:
                if not matched_doc1[i] and not matched_doc2[j]:
                    filtered_pairs.append((i, j, sim))

            if filtered_pairs:
                print(f"  → {len(filtered_pairs)} candidates after filtering")

                llm_matches = 0
                max_workers = 5

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_pair = {
                        executor.submit(
                            self.compare_with_llm,
                            doc1_clauses[i]['text'],
                            doc2_clauses[j]['text']
                        ): (i, j, sim)
                        for i, j, sim in filtered_pairs
                    }

                    for future in as_completed(future_to_pair):
                        i, j, sim = future_to_pair[future]
                        if matched_doc1[i] or matched_doc2[j]:
                            continue
                        try:
                            result = future.result()
                            if result.get('match', False) and result.get('confidence', 0) > 0.7:
                                matched_doc1[i] = True
                                matched_doc2[j] = True
                                match_pairs.append((i, j, sim, result.get('reason', 'LLM match')))
                                llm_matches += 1
                                print(f"  ✅ LLM matched: Doc1[{i}] ↔ Doc2[{j}] (sim: {sim:.3f})")
                        except Exception as e:
                            print(f"  ❌ LLM error for pair ({i}, {j}): {e}")

                print(f"  → {llm_matches} LLM matches found")

        # ------------------------------------------------------------
        # STEP 5: Build result structures
        # ------------------------------------------------------------
        match_map = {i: (j, sim, reason) for i, j, sim, reason in match_pairs}

        matching_details = []
        only_in_doc1 = []

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
                        'confidence': 1.0 if 'high' in reason.lower() or 'match' in reason.lower() else 0.9,
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

        # Compute counts
        llm_matches = sum(1 for _, _, _, reason in match_pairs if 'LLM' in reason)
        high_matches = len(match_pairs) - llm_matches

        print(f"✅ Final matches: {len(match_pairs)} total ({high_matches} via high-sim, {llm_matches} via LLM)")
        print(f"⏱️  Total time: {elapsed:.2f}s")

        return {
            'only_in_doc1': only_in_doc1,
            'only_in_doc2': only_in_doc2,
            'matching_details': matching_details,
            'total_doc1': n1,
            'total_doc2': n2,
            'matching_count': len(match_pairs),
            'processing_time': elapsed,
            'similarity_threshold': similarity_threshold,
            'high_similarity_threshold': high_similarity_threshold,
            'llm_matches': llm_matches,
            'high_sim_matches': high_matches,
            'doc2_best_similarities': [0.0] * n2
        }