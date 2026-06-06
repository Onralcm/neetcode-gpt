from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        char_list = [c for c in corpus]
        merge = []
        for i in range(num_merges):
            freq= {}
            for i in range(len(char_list) - 1):
                if (char_list[i], char_list[i+1]) not in freq:
                    freq[(char_list[i], char_list[i+1])] = 0
                freq[(char_list[i], char_list[i+1])]+=1

            mx = 0
            mx_pair = ('', '')
            for pair, count in freq.items():
                if count > mx:
                    mx = count
                    mx_pair = pair
                elif count == mx and pair < mx_pair:
                    mx_pair = pair

            tmp_list = []
            i = 0
            while i < len(char_list) - 1:
                if char_list[i] == mx_pair[0] and char_list[i+1] == mx_pair[1]:
                    tmp_list.append(char_list[i]+char_list[i+1])
                    i+=1
                else:
                    tmp_list.append(char_list[i])
                
                i+=1

            char_list = tmp_list

            merge.append(mx_pair)

        return merge

