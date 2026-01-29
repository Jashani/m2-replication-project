import pandas as pd
import numpy as np
import os
import random
from scipy.stats import norm

def simulate_m2_experiment(output_directory, num_participants=200):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    TOTAL_TRIALS = 270
    TRIALS_PER_TYPE = int(TOTAL_TRIALS / 2)
    # Targeted performance from supplemental material
    INTACT_D = 0.55
    SCRAMBLED_D = 0.55
    
    # Probabilities for 2AFC
    p_intact = norm.cdf(INTACT_D / np.sqrt(2))
    p_scrambled = norm.cdf(SCRAMBLED_D / np.sqrt(2))

    for p_id in range(1, num_participants + 1):       
        obj_success = 0
        obj_fail = 0
        scr_success = 0
        scr_fail = 0
        
        # Participant variance: Skill and Speed
        p_skill_bias = np.random.normal(0, 0.2)
        p_speed_base = np.random.uniform(600, 1200)
        
        # 3% of participants are "low performers" (< 0 d' overall)
        is_low_performer = np.random.random() < 0.03
        
        trial_types = ["OBJECT"] * TRIALS_PER_TYPE + ["SCRAMBLED"] * TRIALS_PER_TYPE
        random.shuffle(trial_types)
        
        results = []
        time_accumulator = 5000 # Start time offset
        
        obj_counter = 1
        scr_counter = 1

        for i, t_type in enumerate(trial_types):
            # --- 1. RSVP Phase ---
            # Mimicking URL progression in batches of 4
            if t_type == "OBJECT":
                target_id = random.randint(obj_counter, obj_counter + 3)
                recency_boost = 0.10 if (target_id % 4 == 0) else 0.0
                base_url = "https://raw.githubusercontent.com/Jashani/m2-replication-project/refs/heads/main/color_objects/obj"
                foil_url = f"https://raw.githubusercontent.com/Jashani/m2-replication-project/refs/heads/main/foil_objects/obj{target_id}.jpg"
                obj_counter += 4
                prob = p_intact + p_skill_bias + recency_boost
            else:
                target_id = random.randint(scr_counter, scr_counter + 3)
                recency_boost = 0.10 if (target_id % 4 == 0) else 0.0
                base_url = "https://raw.githubusercontent.com/Jashani/m2-replication-project/refs/heads/main/color_scrambled/obj"
                foil_url = f"https://raw.githubusercontent.com/Jashani/m2-replication-project/refs/heads/main/foil_scrambled/obj{target_id}.jpg"
                scr_counter += 4
                prob = p_scrambled + p_skill_bias + recency_boost

            target_img_url = f"{base_url}{target_id}.jpg"
            
            # RSVP Row
            time_accumulator += 2000 # Length of RSVP sequence
            results.append({
                "rt": "", "trial_type": "rsvp", "trial_index": i*2,
                "time_elapsed": time_accumulator, "success": "", "items": "",
                "order": "", "key_response": "", "correct": ""
            })

            # --- 2. 2AFC Phase ---
            # Determine accuracy
            if is_low_performer: prob = 0.50 
            is_correct = np.random.random() < prob
            if is_correct:
                if t_type == "OBJECT": obj_success += 1
                else: scr_success += 1
            else:
                if t_type == "OBJECT": obj_fail += 1
                else: scr_fail += 1
            
            # Logic for order and key_response
            # order "0,1": Left is Target, Right is Foil. Correct key = 37.
            # order "1,0": Left is Foil, Right is Target. Correct key = 39.
            side_choice = random.choice(["0,1", "1,0"])
            if side_choice == "0,1":
                items = f"{target_img_url},{foil_url}"
                key_response = 37 if is_correct else 39
            else:
                items = f"{foil_url},{target_img_url}"
                key_response = 39 if is_correct else 37
            
            # Response Time (Ex-Gaussian)
            rt = np.random.normal(p_speed_base, 150) + np.random.exponential(300)
            rt = max(180, rt) # Study exclusion floor is 200ms
            time_accumulator += int(rt)

            results.append({
                "rt": round(rt, 2),
                "trial_type": "2afc-keyboard",
                "trial_index": (i*2) + 1,
                "time_elapsed": time_accumulator,
                "success": "",
                "items": items,
                "order": side_choice,
                "key_response": key_response,
                "correct": str(is_correct).lower()
            })

        df = pd.DataFrame(results)
        df.to_csv(os.path.join(output_directory, f"p_{p_id:03d}_data.csv"), index=False)
        print(f"Generated p_{p_id:03d}_data.csv: +obj={obj_success}, -obj={obj_fail}, +scr={scr_success}, -scr={scr_fail}")
        

    print(f"Generated 200 files in {output_directory}")

# Execution
# simulate_m2_experiment("simulated_results")

output_dir = 'simulated_data/bad'

if __name__ == "__main__":
    simulate_m2_experiment(output_dir)
