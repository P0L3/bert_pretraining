import pandas as pd
import re
from rapidfuzz import fuzz, process
from tqdm.auto import tqdm
from joblib import Parallel, delayed # For easy parallelization
import collections
import os
import time


# --- Your helper functions remain the same. They are already good. ---
def fix_title(title):
    if not isinstance(title, str): return ""
    if "  " in title:
        placeholder = "@@@"
        title_fixed = re.sub(r'\s{2,}', placeholder, title)
        title_fixed = title_fixed.replace(' ', '')
        title = title_fixed.replace(placeholder, ' ')
    title = title.lower()
    title = re.sub(r'[^\w\s]', '', title) # Your original, good function
    title = " ".join(title.split())
    return title

def extract_title_from_reference(ref_string):
    if not isinstance(ref_string, str): return None
    cleaned_ref = re.sub(r'\[.*?\]', '', ref_string).strip()
    parts = cleaned_ref.split('.')
    potential_titles = [part.strip() for part in parts if len(part.strip()) > 10]
    if not potential_titles: return None
    title = max(potential_titles, key=len)
    return title

def extract_year(date_string):
    if not isinstance(date_string, str): return None
    published_match = re.search(r'Published:.*?(\b(19|20)\d{2}\b)', date_string)
    if published_match: return int(published_match.group(1))
    all_years = re.findall(r'\b(?:19|20)\d{2}\b', date_string)
    if all_years: return int(all_years[-1])
    return None

# --- NEW: The core matching logic, now designed for parallel chunks ---
def find_matches_for_chunk(unique_refs_chunk, yearly_corpus, corpus_titles_map, threshold=85, year_tol=2):
    """
    This function will be run on each CPU core.
    It takes a chunk of unique references and finds matches for them.
    """
    match_map = {}
    for norm_ref_title, ref_year in unique_refs_chunk:
        # Tier 1: Exact Match (Fastest)
        if norm_ref_title in corpus_titles_map:
            match_map[norm_ref_title] = (corpus_titles_map[norm_ref_title], 100)
            continue

        # Prepare candidates using the pre-grouped dictionary
        candidate_titles = []
        candidate_indices = []
        if ref_year is not None:
            for y in range(ref_year - year_tol, ref_year + year_tol + 1):
                if y in yearly_corpus:
                    titles, indices = yearly_corpus[y]
                    candidate_titles.extend(titles)
                    candidate_indices.extend(indices)
        
        # If no candidates found by year, search the whole corpus (or skip if you prefer)
        if not candidate_titles:
             titles, indices = yearly_corpus['all']
             candidate_titles.extend(titles)
             candidate_indices.extend(indices)

        if not candidate_titles:
            continue

        # Tier 2: Fuzzy Match
        best_match = process.extractOne(norm_ref_title, candidate_titles, scorer=fuzz.WRatio, score_cutoff=threshold)

        if best_match:
            matched_title_str, score, original_list_index = best_match
            corpus_idx = candidate_indices[original_list_index]
            match_map[norm_ref_title] = (corpus_idx, score)
    
    return match_map


# --- Main Execution Block ---
if __name__ == '__main__':
    start_time = time.time()
    # Load your data
    df = pd.read_pickle("DATASET/ED4RE_2503/ED4RE_2603.pickle")
    is_list_mask = df['References'].apply(lambda x: isinstance(x, list))
    df_lists_only = df[is_list_mask]
    df = df_lists_only.copy()

    # --- Corpus Preparation (Same as before) ---
    print("Step 1: Preparing the corpus...")
    tqdm.pandas(desc="Normalizing Titles")
    df['normalized_title'] = df['Title'].progress_apply(fix_title)
    tqdm.pandas(desc="Extracting Years")
    df['year'] = df['Date'].progress_apply(extract_year).astype('Int64') # Use nullable integer
    
    print("Creating title-to-index map for exact matches...")
    corpus_titles_map = {title: idx for idx, title in df['normalized_title'].items() if title}
    
    # --- OPTIMIZATION 1: BATCH & DEDUPLICATE ---
    print("\nStep 2: Collecting and deduplicating all references...")
    unique_references_to_match = set()
    for references in tqdm(df['References'], desc="Scanning References"):
        if not isinstance(references, list):
            continue
        for ref_string in references:
            raw_title = extract_title_from_reference(ref_string)
            norm_title = fix_title(raw_title)
            if norm_title:
                ref_year = extract_year(ref_string)
                unique_references_to_match.add((norm_title, ref_year))
    
    unique_references_list = list(unique_references_to_match)
    print(f"Found {len(df['References'].explode())} total references, but only {len(unique_references_list)} unique titles to match.")

    # --- OPTIMIZATION 2: PRE-GROUP CORPUS BY YEAR ---
    print("\nStep 3: Pre-grouping corpus titles by year for fast lookups...")
    yearly_corpus = collections.defaultdict(lambda: ([], []))
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Grouping by Year"):
        if row['normalized_title']:
            # Store as (list_of_titles, list_of_indices)
            yearly_corpus[row['year']][0].append(row['normalized_title'])
            yearly_corpus[row['year']][1].append(idx)
    # Also create an 'all' key for fallback
    yearly_corpus['all'] = (df['normalized_title'].tolist(), df.index.tolist())


    # --- OPTIMIZATION 3: PARALLELIZE THE MATCHING ---
    print(f"\nStep 4: Starting parallel matching on {len(unique_references_list)} unique titles...")
    n_jobs = -1  # Use all available CPU cores
    chunk_size = len(unique_references_list) // (n_jobs if n_jobs != -1 else os.cpu_count()) +1
    chunks = [unique_references_list[i:i + chunk_size] for i in range(0, len(unique_references_list), chunk_size)]
    
    # The parallel execution magic!
    results_list = Parallel(n_jobs=n_jobs)(
        delayed(find_matches_for_chunk)(chunk, yearly_corpus, corpus_titles_map) 
        for chunk in tqdm(chunks, desc="Processing Chunks")
    )
    
    print("\nStep 5: Combining results from parallel jobs...")
    final_match_map = {}
    for res_map in results_list:
        final_match_map.update(res_map)

    # --- FINAL FAST LINKING LOOP ---
    print("\nStep 6: Creating final citation links using the pre-computed map...")
    citation_links = []
    for source_index, row in tqdm(df.iterrows(), total=len(df), desc="Final Linking"):
        references = row['References']
        if not isinstance(references, list): continue
        for ref_string in references:
            norm_ref_title = fix_title(extract_title_from_reference(ref_string))
            if norm_ref_title in final_match_map:
                target_index, score = final_match_map[norm_ref_title]
                citation_links.append({
                    'source_index': source_index,
                    'target_index': target_index,
                    'match_score': score
                })

    # --- Final Results ---
    print(f"\nProcess complete. Found {len(citation_links)} potential links.")
    links_df = pd.DataFrame(citation_links)
    print("\nSample of created links:")
    print(links_df.head())
    print("\nDistribution of match scores:")
    print(links_df['match_score'].describe())
    
    links_df.to_pickle(f"citation_links_{len(links_df)}.pickle")
    elapsed_time = time.time() - start_time
    print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))