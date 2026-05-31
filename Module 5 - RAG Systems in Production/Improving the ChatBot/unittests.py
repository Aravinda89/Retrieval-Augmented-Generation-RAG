import os
import re
import time
import together
from dlai_grader.grading import test_case, print_feedback
from types import FunctionType
import numpy as np
import joblib
import pandas as pd 


def test_check_if_faq_or_product(learner_func):
    def g():
        function_name = learner_func.__name__
        cases = []
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{learner_func} has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)
        query = "What are your working hours?"
        try:
            output, total_tokens = learner_func(query, simplified = True)
        except Exception as e:
            t.failed = True
            t.msg = f"{function_name} raised an exception with the following parameters query = {query}"
            t.want = f"{function_name} must run without exceptions"
            t.got = str(e)
            return [t]
        cases.append(t)
            
        
        t = test_case()
        if not isinstance(output, str):
            t.failed = True
            t.msg = f"{function_name} have wrong type"
            t.want = str
            t.got = type(output)
            return [t]
        cases.append(t)
        t = test_case()
        if len(output.split()) != 1:
            t.failed = True
            t.msg = f"Output must have only one word"
            t.want = "One element"
            t.got = output      
        cases.append(t)
        
        t = test_case()
        max_tokens = 180
        if total_tokens > max_tokens:
            t.failed = True
            t.msg = f"Total tokens for query={query} exceeded {max_tokens} tokens"
            t.want = f"At most {max_tokens} tokens in the answer"
            t.got = total_tokens
        cases.append(t)
        
        
        queries = ['What is your return policy?', 
           'Give me three examples of blue Tshirts you have available.', 
           'How can I contact the user support?', 
           'Do you have blue Dresses?',
           'Create a look suitable for a wedding party happening during dawn.']

        labels = ['FAQ', 'Product', 'FAQ', 'Product', 'Product']
        
        for query, correct_label in zip(queries, labels):
            t = test_case()
            response, total_tokens = learner_func(query, simplified = True)
            label = response
            if label != correct_label:
                t.failed = True
                t.msg = f'Incorrect label for query={query}'
                t.want = correct_label
                t.got = response
            cases.append(t)
            t = test_case()
            max_tokens = 180
            if total_tokens > max_tokens:
                t.failed = True
                t.msg = f"Total tokens for query={query} exceeded {max_tokens} tokens"
                t.want = f"At most {max_tokens} tokens in the answer"
                t.got = total_tokens
            cases.append(t)
    

        return cases
    
        
    
    

    

    cases = g()
    print_feedback(cases)

def test_query_on_faq(learner_func):
    def g():
        function_name = learner_func.__name__
        cases = []
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{learner_func} has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        query = "What are your working hours?"
        try:
            output = learner_func(query, simplified = True)
        except Exception as e:
            t.failed = True
            t.msg = f"{function_name} raised an exception with the following parameters query = {query}"
            t.want = f"{function_name} must run without exceptions"
            t.got = str(e)
            return [t]
            
        t = test_case()
        if not isinstance(output, dict):
            t.failed = True
            t.msg = f"{function_name} have wrong type"
            t.want = str
            t.got = type(output)
            return [t]
        cases.append(t)
        import re


        pattern = r'\bQuestion\b'

        matches = re.findall(pattern, output['prompt'])
        t = test_case()
        if len(matches) > 6:
            t.failed = True
            t.msg = "There are more than 5 questions in your prompt"
            t.want = "Exactly 5 questions must be contained in your prompt"
            t.got = f"{len(matches) - 1} questions"
        cases.append(t)
        return cases

    cases = g()
    print_feedback(cases)


def test_decide_task_nature(learner_func):
    def g():
        function_name = learner_func.__name__
        cases = []
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{learner_func} has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        query = "What are your working hours?"
        try:
            output, total_tokens = learner_func(query)
        except Exception as e:
            t.failed = True
            t.msg = f"{function_name} raised an exception with the following parameters query = {query}"
            t.want = f"{function_name} must run without exceptions"
            t.got = str(e)
            return [t]
            
        t = test_case()
        if not isinstance(output, str):
            t.failed = True
            t.msg = f"{function_name} have wrong type"
            t.want = str
            t.got = type(output)
            return [t]
        cases.append(t)
        
        
        queries = ["Give me two sneakers with vibrant colors.",
                   "What are the most expensive clothes you have in your catalogue?",
                   "I have a green Dress and I like a suggestion on an acessory to match with it.",
                   "Give me three trousers with vibrant colors you have in your catalogue.",
                   "Create a look for a woman walking in a park on a sunny day. It must be fresh due to hot weather."
                   ]

        labels = ['technical', 'technical', 'creative', 'technical', 'creative']
        
        error_count = 0
        for query, correct_label in zip(queries, labels):
            t = test_case()
            response, total_tokens = learner_func(query, simplified = True)
            label = response
            if not label == correct_label:
                error_count += 1
            cases.append(t)
            t = test_case()
            if total_tokens > 150:
                t.failed = True
                t.msg = f"Incorrect token count for query = {query}"
                t.want = "<150"
                t.got = total_tokens
            cases.append(t)
        t = test_case()
        if error_count > 1:
            t.failed = True
            t.msg = f"Error count greater than 80%"
            t.want = f"Error count < 80%"
            t.got = f"Error count = {100*error_count/len(queries):.2f}"
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def test_get_params_for_task(learner_func):
    def g():
        function_name = learner_func.__name__
        cases = []
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{learner_func} has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)
        task = 'technical'
        t = test_case()

        try:
            output = learner_func(task)
        except Exception as e:
            t.failed = True
            t.msg = f"{function_name} raised an exception with the following parameters task = {task}"
            t.want = f"{function_name} must run without exceptions"
            t.got = str(e)
            return [t]
        cases.append(t)
            
        t = test_case()
        if not isinstance(output, dict):
            t.failed = True
            t.msg = f"{function_name} have wrong type"
            t.want = dict
            t.got = type(output)
            return [t]
        cases.append(t)
        return cases


    cases = g()
    print_feedback(cases)

    
def test_get_relevant_products_from_query(learner_func):
    def g():
        function_name = learner_func.__name__
        cases = []
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{learner_func} has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)
        query = "Give me three Tshirts to use in sunny days"
        try:
            output, total_tokens = learner_func(query, simplified = True)
        except Exception as e:
            t.failed = True
            t.msg = f"{function_name} raised an exception with the following parameters query = {query}"
            t.want = f"{function_name} must run without exceptions"
            t.got = str(e)
            return [t]
        cases.append(t)
        
        
        ids = set([t.properties['product_id'] for t in output])
        ids_solution = set([3328, 35983, 54935, 6939, 33565, 49964, 2863, 2866, 1844, 1845, 1846, 1847, 1853, 9539, 1866, 4298, 1867, 3431, 37608, 3318])
        t = test_case()
        if ids != ids_solution:
            t.failed = True
            t.msg = f"Incorrect result for query = {query} and simplified = True"
            t.want = f"Product IDs must be {ids_solution}"
            t.got = f"Product IDs output are: {ids}"
        cases.append(t)
        
        return cases
        
    cases = g()
    print_feedback(cases)

    
columns = ['query', 'result', 'total_tokens', 'kwargs']
logging_dataset = pd.DataFrame(columns=columns)


kwargs = {'prompt': "You will be provided with an FAQ for a cloth store. \n    Answer the instruction based on it. You might use more than one question and answer to make your answer. Only answer the question and do not mention that you have access to a FAQ. \n    <scratchpad>\n    PROVIDED FAQ: Question: Can I return a sale item? Answer: Sale items are final sale and cannot be returned or exchanged, unless stated otherwise. Type: returns and exchanges\nQuestion: How long does it take to process a return? Answer: Return processing typically takes 5-7 business days from when the item is received at our warehouse. Type: returns and exchanges\nQuestion: How do I exchange an item? Answer: Initiate an exchange through our Returns Center, selecting the item you wish to exchange and the desired replacement. Type: returns and exchanges\nQuestion: Are return shipping costs covered? Answer: We provide a prepaid return label for domestic returns. For international returns, shipping is at the customer's cost. Type: returns and exchanges\nQuestion: What is your return policy timeframe? Answer: We accept returns within 30 days of delivery. Conditions apply for specific categories like accessories. Type: returns and exchanges\n\n    </scratchpad>\n    Question: What is your return policy?\n        ",
 'role': 'user',
 'temperature': 1.0,
 'top_p': 1.0,
 'max_tokens': 500,
 'model': 'Qwen/Qwen3.5-9B'}

query = "What is your return policy?"

result = {'role': 'assistant',
 'content': 'Our return policy allows you to return or exchange an item within 30 days of delivery. However, there are some exceptions to this policy, particularly for specific categories of items such as accessories.\n\nAdditionally, sale items are final sale and cannot be returned or exchanged, unless stated otherwise.\n\nFor returns and exchanges, please note the following:\n\n- Return shipping costs will be covered for domestic returns.\n- International returns will incur the shipping cost, as the customer is responsible for paying for this.\n- Return processing typically takes 5-7 business days from when the item is received at our warehouse.\n- To initiate an exchange, please use our Returns Center, where you can select the item you wish to exchange and the desired replacement.',
 'total_tokens': 439}

total_tokens = 112

def test_generate_log(learner_func):
    def g():
        function_name = learner_func.__name__
        cases = []
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{learner_func} has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)
        query = "Give me three Tshirts to use in sunny days"
        try:
            output = learner_func(query, kwargs, total_tokens, result, logging_dataset)
        except Exception as e:
            t.failed = True
            t.msg = f"{function_name} raised an exception."
            t.want = f"{function_name} must run without exceptions"
            t.got = str(e)
            return [t]
        cases.append(t)
        
        t = test_case()
        if output is not None:
            t.failed = True
            t.msg = "The function must have no return value"
            t.want = "Function must return None"
            t.got = output
        cases.append(t)
            
        
        
        t = test_case()
        if logging_dataset['query'][0] != query:
            t.failed = True
            t.msg = "Incorrect value for query column"
            t.want = query
            t.got = logging_dataset['query']   
        cases.append(t)
        
        t = test_case()
        if logging_dataset['total_tokens'][0] != total_tokens + result['total_tokens']:
            t.failed = True
            t.msg = "Incorrect value for total_tokens column. Make sure to add total_tokens + result['total_tokens'] in the columns"
            t.want = total_tokens + result['total_tokens']
            t.got = logging_dataset['total_tokens'] 
        cases.append(t)
        
        return cases
        
    cases = g()
    print_feedback(cases)
