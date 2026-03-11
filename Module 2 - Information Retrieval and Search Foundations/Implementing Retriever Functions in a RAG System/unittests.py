from dlai_grader.grading import test_case, print_feedback
from types import FunctionType
import utils


data = utils.read_dataframe("news_data_dedup.csv")



def test_bm25_retrieve(learner_func):
    def g():
        cases = []
        func_name = learner_func.__name__
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]

        top_k = 3
        test_query = "Should I invest in startups?"
        
        top_k_indices = learner_func(test_query, top_k)

        t = test_case()
        if not isinstance(top_k_indices, list):
            t.failed = True
            t.msg = "output has wrong type"
            t.want = list
            t.got = type(top_k_indices)
            return [t]

        t = test_case()
        if len(top_k_indices) != top_k:
            t.failed = True
            t.msg = f"output has wrong length when using top_k={top_k}"
            t.want = top_k
            t.got = len(top_k_indices)
            return [t]

        expected_indices = [863, 848, 716]

        t = test_case()
        if expected_indices != top_k_indices:
            t.failed = True
            t.msg = "Incorrect output indices"
            t.want = expected_indices
            t.got = top_k_indices
        cases.append(t)

        top_k = 10
        test_query = "Should I invest in startups?"
        
        top_k_indices = learner_func(test_query, top_k)

        t = test_case()
        if not isinstance(top_k_indices, list):
            t.failed = True
            t.msg = "top_k_indices has wrong type"
            t.want = list
            t.got = type(top_k_indices)
            return [t]

        t = test_case()
        if len(top_k_indices) != top_k:
            t.failed = True
            t.msg = f"top_k_indices has wrong length when using top_k={top_k}"
            t.want = top_k
            t.got = len(top_k_indices)
            return [t]

        return cases

    cases = g()
    print_feedback(cases)

def test_semantic_search_retrieve(learner_func, EMBEDDING):
    def g():
        cases = []
        func_name = learner_func.__name__
    
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]

        top_k = 3
        test_query = "Should I invest in startups?"
        
        top_k_indices = learner_func(test_query, top_k)

        t = test_case()
        if not isinstance(top_k_indices, list):
            t.failed = True
            t.msg = "top_k_indices has wrong type"
            t.want = list
            t.got = type(top_k_indices)
            return [t]

        t = test_case()
        if len(top_k_indices) != top_k:
            t.failed = True
            t.msg = f"top_k_indices has wrong length when using top_k={top_k}"
            t.want = top_k
            t.got = len(top_k_indices)
            return [t]

        expected_indices = [863, 416, 624]

        t = test_case()
        if expected_indices != top_k_indices:
            t.failed = True
            t.msg = "Incorrect indices in 'top_k_indices'"
            t.want = expected_indices
            t.got = top_k_indices
        cases.append(t)

        top_k = 10
        test_query = "Should I invest in startups?"
        
        top_k_indices = learner_func(test_query, top_k)

        t = test_case()
        if not isinstance(top_k_indices, list):
            t.failed = True
            t.msg = "top_k_indices has wrong type"
            t.want = list
            t.got = type(top_k_indices)
            return [t]

        t = test_case()
        if len(top_k_indices) != top_k:
            t.failed = True
            t.msg = f"top_k_indices has wrong length when using top_k={top_k}"
            t.want = top_k
            t.got = len(top_k_indices)
            return [t]

        return cases

    cases = g()
    print_feedback(cases)




def test_reciprocal_rank_fusion(learner_func):
    def g():
        cases = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "reciprocal_rank_fusion has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]

        top_k = 10
        l1 = [17, 29, 28, 26, 18, 14, 1, 0, 16, 11]
        l2 = [17, 26, 16, 25, 18, 24, 13, 11, 6, 12]
        
        top_k_indices = learner_func(l1, l2, top_k)

        t = test_case()
        if not isinstance(top_k_indices, list):
            t.failed = True
            t.msg = "top_k_indices has wrong type"
            t.want = list
            t.got = type(top_k_indices)
            return [t]

        t = test_case()
        if len(top_k_indices) != top_k:
            t.failed = True
            t.msg = f"top_k_indices has wrong length when using top_k={top_k}"
            t.want = top_k
            t.got = len(top_k_indices)
            return [t]

        expected_indices = [17, 26, 18, 16, 11, 29, 28, 25, 14, 24]

        t = test_case()
        if expected_indices != top_k_indices:
            t.failed = True
            t.msg = "Incorrect indices in 'top_k_indices'"
            t.want = expected_indices
            t.got = top_k_indices
        cases.append(t)

        top_k = 4
        test_query = "Should I invest in startups?"
        
        top_k_indices = learner_func(l1, l2, top_k)

        t = test_case()
        if not isinstance(top_k_indices, list):
            t.failed = True
            t.msg = "top_k_indices has wrong type"
            t.want = list
            t.got = type(top_k_indices)
            return [t]

        t = test_case()
        if len(top_k_indices) != top_k:
            t.failed = True
            t.msg = f"top_k_indices has wrong length when using top_k={top_k}"
            t.want = top_k
            t.got = len(top_k_indices)
            return [t]

        return cases

    cases = g()
    print_feedback(cases)



def exercise_5(learner_func):
    def g():
        cases = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "recall has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]

        l1 = [1,2,4]
        l2 = [1,2,3,4]
        
        recall_score = learner_func(l1, l2)

        t = test_case()
        if not isinstance(recall_score, float):
            t.failed = True
            t.msg = "recall score has wrong type"
            t.want = float
            t.got = type(recall_score)
            return [t]


        t = test_case()
        if recall_score != 0.75:
            t.failed = True
            t.msg = "Incorrect recall for sample list"
            t.want = 0.75
            t.got = recall_score
        cases.append(t)

        l1 = [1,2,3,4,5]
        l2 = [1,2,3,4,5]
        
        recall_score = learner_func(l1, l2)

        t = test_case()
        if not isinstance(recall_score, float):
            t.failed = True
            t.msg = "recall score has wrong type"
            t.want = float
            t.got = type(recall_score)
            return [t]


        t = test_case()
        if recall_score != 1.:
            t.failed = True
            t.msg = "Incorrect recall for sample list"
            t.want = 1.
            t.got = recall_score
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)