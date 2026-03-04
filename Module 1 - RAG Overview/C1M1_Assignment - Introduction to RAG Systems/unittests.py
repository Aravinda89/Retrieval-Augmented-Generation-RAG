from dlai_grader.grading import test_case, print_feedback
from types import FunctionType
import pandas as pd

NEWS_DATA = pd.read_csv("./news_data_dedup.csv").to_dict(orient = 'records')

def query_by_index(list_of_indices, dataset):
    """
    Retrieves elements from a dataset based on specified indices.

    Parameters:
    list_of_indices (list of int): A list containing the indices of the desired elements in the dataset.
    dataset (list or sequence): The dataset from which elements are to be retrieved. It should support indexing.

    Returns:
    list: A list of elements from the dataset corresponding to the indices provided in list_of_indices.
    """
    output = []
    for i in list_of_indices:
        output.append(dataset[i])

    return output

def test_format_relevant_data(learner_func):
    def g():
        func_name = learner_func.__name__
        cases = []
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        relevant_data = NEWS_DATA[5:9]
        res = learner_func(relevant_data).lower()
        necessary_keywords = ['title', 'url', 'published', 'description']
        
        for keyword in necessary_keywords:
            t = test_case()
            if keyword not in res:
                t.failed = True
                t.msg = f"Keyword {keyword.capitalize()} not present in your prompt"
                t.want = f"Keyword {keyword.capitalize()} must be in the prompt"
            cases.append(t)
            t = test_case()
            if keyword in ['title','url','published', 'description']:
                number_occurrences = res.count(keyword)
                if number_occurrences != len(relevant_data):
                    t.failed = True
                    t.want = f"There must be {len(relevant_data)} occurrences of {keyword} by calling augment_prompt with relevant_data = NEWS_DATA[5:9]"
                    t.got = f"Number of occurrences of {keyword}: {number_occurrences}"
                    t.want = f"{len(relevant_data)} occurrences"
                cases.append(t)
        return cases

    cases = g()
    print_feedback(cases)


def test_get_relevant_data(learner_func):
    def g():
        func_name = learner_func.__name__
        cases = []
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
    
        t = test_case()
        query = "This is a test query"
        top_k = 3
        result =[{'guid': 'e78d129bee161f6416d20ab0ae66f5a9',
  'title': 'UN human rights chief ‘horrified’ by reports of mass graves at Gaza hospitals',
  'description': 'Mass graves with hundreds of bodies have reportedly been uncovered at hospitals raided by Israeli troops Read Full Article at RT.com',
  'venue': 'RT',
  'url': 'https://www.rt.com/news/596463-un-mass-graves-gaza-hospitals/?utm_source=rss&utm_medium=rss&utm_campaign=RSS',
  'published_at': '2024-04-23',
  'updated_at': '2024-04-27'},
 {'guid': '79c0f5715f341c65c0d9abd4890f35c0',
  'title': '‘Trust your gut’: Terrifying Airbnb discovery',
  'description': 'A mum has shared her disturbing experience staying in an Airbnb with her four teen daughters - who she believes were being watched by someone with a sinister plan.',
  'venue': 'News.com.au',
  'url': 'https://www.news.com.au/world/north-america/mum-teen-girls-forced-to-leave-their-airbnb-after-unsettling-discovery-trust-your-gut/news-story/2ffe4b9152c6080840e6200a21e9c830?from=rss-basic',
  'published_at': '2024-04-27',
  'updated_at': '2024-04-27'},
 {'guid': '2de17d633142978a5409df1445ad538c',
  'title': 'Basic Materials Roundup: Market Talk',
  'description': 'BASF, Fortescue and more in the latest Market Talks covering Basic Materials.',
  'venue': 'WSJ',
  'url': 'https://www.wsj.com/articles/basic-materials-roundup-market-talk-14e6ab07',
  'published_at': '2024-04-26',
  'updated_at': '2024-04-26'}]
        guid_result = set([d['guid'] for d in result])
        try:
            output = learner_func(query, top_k = 3)
        except Exception as e:
            t.failed = True
            t.msg = f"{learner_func} raised an exception for query = {query} and top_k = {top_k}"
            t.got = f"Exception: {e}"
            return [t]
        t = test_case()
        if not isinstance(output, list):
            t.failed = True
            t.msg = f"Incorrect output type"
            t.want = list
            t.got = type(output)
            return [t]
        
        t = test_case()
        if len(output) != top_k:
            t.failed = True
            t.msg = f"Incorrect number of retrieved documents for top_k = {3}"
            t.want = top_k
            t.got = len(output)
        cases.append(t)
        t = test_case()
        try:
            output_guid = set([d['guid'] for d in output])
        except Exception as e:
            t.failed = True
            t.msg = f"Couldn't extract guid from your solution"
            t.want = "Each element of output must habe the guid key"
            t.got = f"Exception thrown: {e}"
            return [t]
        if output_guid != guid_result:
            t.failed = True
            t.msg = f"Incorrect retrieved documents for query = {query} and top_k = {top_k}"
            t.want = f"Guid of retrieved documents: {guid_result}"
            t.got = output_guid
        cases.append(t)
        return cases

    cases = g()
    print_feedback(cases)
        


