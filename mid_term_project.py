from mlxtend.frequent_patterns import fpgrowth, association_rules, apriori
import pandas as pd
import numpy as np
from itertools import combinations
import time


class AssociationRulesGenerator:
    def __init__(self, min_confidence, min_support):
        self.conf = min_confidence
        self.min_supp = min_support

    def read_data_set(self,transactions_file_path):
        """
        Read CSV file
        Args:
            transactions_file_path (str): Path to the CSV file

        Returns:
            pd.DataFrame: One-hot-encoded DataFrame
        """
        items_transactions_df = pd.read_csv(transactions_file_path)
        items_transactions_df = items_transactions_df.fillna(0)
        return items_transactions_df

    def brute_force(self, items_df):
        """This function returns frequent items using brute force technique

        Args:
            items_df (DataFframe): Items Transactiion dataframe

        Returns:
            Dataframe: Frequent items 
        """
        
        
        # Generate itemsets of size 1 with their support
        fq_item_set1 = {k: round(items_df[k].sum() / len(items_df), 3) for k in items_df.columns}
        freq_item_set_df = pd.DataFrame(list(fq_item_set1.items()), columns=["itemsets", "support"])
        selected_keys = list(fq_item_set1.keys())

        print(f"Item Set 1 \n{freq_item_set_df}")
        item_set_count = 2

        while len(selected_keys) != 0:
            items_combinations = combinations(selected_keys, item_set_count)
            support_val_dict = {}
            for item_key in items_combinations:
                t = np.prod(items_df.loc[:, item_key].values, axis=1)
                support_val_dict[item_key] = round(sum(t == 1) / len(items_df), 3)

            #break if support values dictionary is empty
            if len(support_val_dict) == 0:
                selected_keys = []
                break

            item_set_df = pd.DataFrame(list(support_val_dict.items()), columns=["itemsets", "support"])

            if len(item_set_df[item_set_df['support'] >= self.min_supp]) == 0:
                selected_keys = []
                break
            
            #Printtingeach item set
            print(f"Items Set {item_set_count}\n{item_set_df}")
            freq_item_set_df = pd.concat([freq_item_set_df, item_set_df])
            item_set_count += 1

        #Checking Support
        freq_item_set_df = freq_item_set_df[freq_item_set_df['support'] >= self.min_supp].reset_index(drop=True)
        print(f"Final Item Sets: \n{freq_item_set_df}")
        freq_item_set_df['itemsets'] = [k if isinstance(k, tuple) else (k,) for k in freq_item_set_df.itemsets] 
        return freq_item_set_df
    
    def apriori(self, items_df):
        """
        Apriori algorithm.
        """
        print("Frequency Item Set Apriori Algorithm")
        freq_item_set_df = apriori(items_df, min_support=self.min_supp, use_colnames=True)
        print(f"Final Items Set : \n{freq_item_set_df}")
        return freq_item_set_df

    def fp_growth(self, items_df):
        """
         FP-growth algorithm.
        """
        print("Frequency Item Set using FP-Growth Algorithm")
        freq_item_set_df = fpgrowth(items_df, min_support=self.min_supp, use_colnames=True)
        print(f"Final Items Set: \n{freq_item_set_df}")
        return freq_item_set_df
    
    def comparions(self,bf_t,ap_t,fp_t):
        print(f"Apriori is {bf_t//ap_t} times faster than brute force ")
        print(f"FP-Growth is {bf_t//fp_t} times faster than brute force ")
        print(f"FP-Growth is {ap_t//fp_t} time faster than Apriori")
        


    def rule_miner(self, item_file_path):
        
        data = self.read_data_set(item_file_path)

        print(f"Executing using Brute Force Algorithm")
        bf_st = time.time()
        bruteforce_items_support_df = self.brute_force(data)
        brute_force_rules = association_rules(bruteforce_items_support_df, metric="confidence", min_threshold=self.conf)
        print("Brute Algorithm_rules")
        for i, rule in brute_force_rules.iterrows():
            print(f"Rule {i+1}:  {', '.join(rule['antecedents'])} => {', '.join(rule['consequents'])}; confidence: {round(rule['confidence'], 3)}")
        bf_et = time.time()
        bf_t = bf_et-bf_st
        print(f"Brute_force execution time : {round(bf_t,3)}")
        

        print("Executing Apriori Algorithm")
        ap_st = time.time()
        apriori_items_support_df = self.apriori(data)
        apriori_rules = association_rules(apriori_items_support_df, metric="confidence", min_threshold=self.conf)
        apriori_rules = apriori_rules[['antecedents', 'consequents', 'confidence']]
        print("Apriori Rules")
        for i, rule in apriori_rules.iterrows():
            print(f"Rule {i+1}:  {', '.join(rule['antecedents'])} => {', '.join(rule['consequents'])}; confidence: {round(rule['confidence'], 3)}")
        ap_et = time.time()
        ap_t = ap_et - ap_st
        print(f"Apriori execution time : {round(ap_t,3)}")
        
        print("Executiong FP-growth Algorithm")
        fp_st = time.time()
        fp_growth_items_support_df = self.fp_growth(data)
        fp_growth_rules = association_rules(fp_growth_items_support_df, metric="confidence", min_threshold=self.conf)
        fp_growth_rules = fp_growth_rules[['antecedents', 'consequents', 'confidence']]
        print("FP-Growth Rules:")
        for i, rule in apriori_rules.iterrows():
            print(f"Rule {i+1}:  {', '.join(rule['antecedents'])} => {', '.join(rule['consequents'])}; confidence: {round(rule['confidence'], 3)}")
        fp_et = time.time()
        fp_t = fp_et -fp_st
        print(f"FP-Growth Execution time : {round(fp_t,3)}")

        self.comparions(bf_t=bf_t, ap_t=ap_t, fp_t=fp_t)

def main():
    datasets = {
        1: "Data/Items1.csv",
        2: "Data/Items2.csv",
        3: "Data/Items3.csv",
        4: "Data/Items4.csv",
        5: "Data/Items5.csv"
    }

    for dataset_selection, item_file_path in datasets.items():
        print(f"\nProcessing Dataset {dataset_selection}: {item_file_path}")

        
        min_support = float(input(f"Enter min_threshold_support (in %): ")) / 100
        min_confidence = float(input(f"Enter min_threshold_confidence  (in %): ")) / 100
                
        association_rules_generator = AssociationRulesGenerator(min_confidence, min_support)

        association_rules_generator.rule_miner(item_file_path)
      



if __name__ == "__main__":
    main()
