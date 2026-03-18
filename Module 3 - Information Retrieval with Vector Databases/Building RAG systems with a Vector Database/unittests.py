import os
import re
import time
import together
from dlai_grader.grading import test_case, print_feedback
from types import FunctionType
import numpy as np




def check_object(obj, prop, reference_val, index, function_name ):
    if prop not in obj.keys():
        return (False, f"{prop} not in object. Review the output of {function_name} or double check if you loaded the correct collection.")
    if not (reference_val in obj[prop]):
        return (False, f"{prop} value of {index}-th object returned does not contain {reference_val}")
    return (True, "")


def test_filter_by_metadata(learner_func, client):
    def g():
        cases = []
        function_name = learner_func.__name__
        collection = client.collections.get('bbc_collection')
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{function_name} has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        properties_test = {"title":['US', "China"], "chunk": ['Brazil', 'France']}
        for i, (prop, ref_values) in enumerate(properties_test.items()):
            t = test_case()
            try:
                result = learner_func(metadata_property = prop, collection = collection,  values = ref_values, limit = i+1)
            except Exception as e:
                t.failed = True
                t.msg = f"{function_name} must run without exceptions."
                t.want = f"a list with {i+1} elements, and {prop} containing {ref_values}"
                t.got = f"The following exception whas thrown with parameters: property = {prop}, values = {ref_values}, limit = {i+1}. Exception: {e}"
                return [t]

            for value in ref_values:
                result = learner_func(metadata_property = prop, collection = collection, values = [value], limit = i+1)
                t = test_case()
                if len(result) != i+1:
                    t.failed = True
                    t.msg = f"Incorrect number of elements for function call {function_name}(property = {prop}, values = {[value]}, limit = {i+1})"
                    t.want = i+1
                    t.got = len(result)
                cases.append(t)
                t = test_case()
                for ref_value in ref_values:
                    for obj in  learner_func(metadata_property = prop, collection=collection, values = [ref_value], limit = i+1):
                        passed, msg = check_object(obj, prop, reference_val = ref_value, index = i, function_name = function_name)
                        if not passed:
                            t.failed = True
                            t.msg = msg
                            t.want = f"{function_name}(property = {prop}, values = {[value]}, limit = {i+1}) must return {prop} containing {ref_value}"
                            t.got = f"returned title is {obj.title}"
                        cases.append(t)
        return cases

    cases = g()
    print_feedback(cases) 


def check_object_equal(obj, prop, reference_val, index, function_name ):
    if prop not in obj.keys():
        return (False, f"{prop} not in object. Review the output of {function_name} or double check if you loaded the correct collection.")
    if not (reference_val == obj[prop]):
        return (False, f"{prop} value of {index}-th object returned is different from {reference_val}")
    return (True, "")


def test_semantic_search_retrieve(learner_func, client):
    def g():
        cases = []
        function_name = learner_func.__name__
        collection = client.collections.get('bbc_collection')
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{function_name} has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        properties_test = [{"query":"Conflicts in France", 'results': ["After France's election shock comes the real power struggle", "After France's election shock comes the real power struggle"]},
                                     {"query":"Famous actor marries", 'results':["Sandi Toksvig officiates wedding of Abba's Björn Ulvaeus"]}]
        for i, query in enumerate(properties_test):
            t = test_case()
            try:
                result = learner_func(query = query['query'], collection = collection, top_k = len(query['results']))
            except Exception as e:
                t.failed = True
                t.msg = f"{function_name} must run without exceptions."
                t.want = f"a list with { len(query['results'])} elements."
                t.got = f"The following exception whas thrown with parameters: query = {query}, top_k = {top_k}. Exception: {e}"
                return [t]
            
            top_k = len(query['results'])
            results = query['results']
            query = query['query']
            result = learner_func(query = query, collection = collection,  top_k = top_k)
            t = test_case()
            if len(result) != top_k:
                t.failed = True
                t.msg = f"Incorrect number of elements for function call {function_name}(query = {query},  top_k = {top_k})"
                t.want = top_k
                t.got = len(result)
            cases.append(t)
            t = test_case()
            for obj in  learner_func(query = query, collection=collection, top_k = top_k):
                for title in results:
                    passed, msg = check_object_equal(obj, 'title', reference_val = title, index = i, function_name = function_name)
                    if not passed:
                        t.failed = True
                        t.msg = msg
                        t.want = f"{function_name}(query = '{query}', top_k = {top_k}) must return 'title' equals to {title}"
                        t.got = f"returned title is {obj['title']}"
                    cases.append(t)
        return cases

    cases = g()
    print_feedback(cases) 
    
def test_bm25_retrieve(learner_func, client):
    def g():
        cases = []
        function_name = learner_func.__name__
        collection = client.collections.get('bbc_collection')
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{function_name} has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        properties_test = [{"query":"Conflicts in France", 'results': ["D-Day remembrance planes will be found, says Shapps", "D-Day remembrance planes will be found, says Shapps"]},
                                     {"query":"Famous actor marries", 'results':["Media tycoon Rupert Murdoch marries for fifth time"]}]
        for i, query in enumerate(properties_test):
            t = test_case()
            try:
                result = learner_func(query = query['query'], collection = collection, top_k = len(query['results']))
            except Exception as e:
                t.failed = True
                t.msg = f"{function_name} must run without exceptions."
                t.want = f"a list with { len(query['results'])} elements."
                t.got = f"The following exception whas thrown with parameters: query = {query}, top_k = {top_k}. Exception: {e}"
                return [t]
            
            top_k = len(query['results'])
            results = query['results']
            query = query['query']
            result = learner_func(query = query, collection = collection,  top_k = top_k)
            t = test_case()
            if len(result) != top_k:
                t.failed = True
                t.msg = f"Incorrect number of elements for function call {function_name}(query = {query},  top_k = {top_k})"
                t.want = top_k
                t.got = len(result)
            cases.append(t)
            t = test_case()
            for obj in  learner_func(query = query, collection=collection, top_k = top_k):
                for title in results:
                    passed, msg = check_object_equal(obj, 'title', reference_val = title, index = i, function_name = function_name)
                    if not passed:
                        t.failed = True
                        t.msg = msg
                        t.want = f"{function_name}(query = '{query}', top_k = {top_k}) must return 'title' equals to {title}"
                        t.got = f"returned title is {obj['title']}"
                    cases.append(t)
        return cases

    cases = g()
    print_feedback(cases) 


def test_hybrid_retrieve(learner_func, client):
    def g():
        cases = []
        function_name = learner_func.__name__
        collection = client.collections.get('bbc_collection')
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{function_name} has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        properties_test = [{"query":"Conflicts in France", 'results': ["Whistles and boos at France-Israel football match", "Whistles and boos at France-Israel football match"]},
                                     {"query":"Famous actor marries", 'results':["Media tycoon Rupert Murdoch marries for fifth time","Media tycoon Rupert Murdoch marries for fifth time", "Lana Del Rey reportedly marries alligator tour guide in Louisiana "]}]
        for i, query in enumerate(properties_test):
            t = test_case()
            try:
                result = learner_func(query = query['query'], collection = collection, top_k = len(query['results']))
            except Exception as e:
                t.failed = True
                t.msg = f"{function_name} must run without exceptions."
                t.want = f"a list with { len(query['results'])} elements."
                t.got = f"The following exception whas thrown with parameters: query = {query}, top_k = {top_k}. Exception: {e}"
                return [t]
            
            top_k = len(query['results'])
            results = query['results']
            query = query['query']
            result = learner_func(query = query, collection = collection,  top_k = top_k)
            t = test_case()
            if len(result) != top_k:
                t.failed = True
                t.msg = f"Incorrect number of elements for function call {function_name}(query = {query},  top_k = {top_k})"
                t.want = top_k
                t.got = len(result)
                return [t]
            cases.append(t)
            t = test_case()
            for obj, title in zip(learner_func(query = query, collection=collection, top_k = top_k), results):
                passed, msg = check_object_equal(obj, 'title', reference_val = title, index = i, function_name = function_name)
                if not passed:
                    t.failed = True
                    t.msg = msg
                    t.want = f"{function_name}(query = '{query}', top_k = {top_k}) must return 'title' equals to {title}"
                    t.got = f"returned title is {obj['title']}"
                cases.append(t)
        return cases

    cases = g()
    print_feedback(cases) 


def test_semantic_search_with_reranking(learner_func, client):
    def g():
        cases = []
        function_name = learner_func.__name__
        collection = client.collections.get('bbc_collection')
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{function_name} has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        expected_output = ["The Papers: Israel's 'tragic error' and Labour's 'pro-building' bid",
 'MoT boss says 72-day wait for test is new normal',
 'Pour a proper pint, Trading Standards tells pubs',
 'Our interactive guide to the latest voting trends',
 'Tories need a Budget bounce but can Hunt deliver?']
        # Set a query
        query = 'This is a test query'
        rerank_query = 'This is a test rerank query'
        rerank_property = 'chunk'
        top_k = 5
        t = test_case()
        try:
            result = learner_func(query = query, collection = collection, top_k = top_k, rerank_property = rerank_property, rerank_query = rerank_query )
        except Exception as e:
            t.failed = True
            t.msg = f"{function_name} must run without exceptions."
            t.want = f"a list with { top_k } elements."
            t.got = f"The following exception whas thrown with parameters: query = {query}, rerank_query = {rerank_query}, top_k = {top_k}, rerank_property = {rerank_property}. Exception: {e}"
            return [t]
        
        result = [r['title'] for r in result]

        t = test_case()
        if len(result) != top_k:
            t.failed = True
            t.msg = f"Incorrect number of elements for function call {function_name}( query = {query}, collection = collection,rerank_query = {rerank_query}, top_k = {top_k}, rerank_property = {rerank_property})"
            t.want = top_k
            t.got = len(result)
            return [t]
        cases.append(t)
        t = test_case()
        if result != expected_output:
            t.failed = True
            t.msg = f'Incorrect output for {function_name}(query = {query}, collection = collection,rerank_query = {rerank_query}, top_k = {top_k}, rerank_property = {rerank_property})'
            t.want = expected_output
            t.got = result
        cases.append(t)
        return cases
    cases = g()
    print_feedback(cases) 



