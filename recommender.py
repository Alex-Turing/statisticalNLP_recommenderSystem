from csv import DictReader
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from random import shuffle, sample
import sys

NUM_RECS = 101 
MAX_NEW_RECS = 8
isInit = True

def load_articles(file_path, file_format, num=None):
    """
    Load articles from a specified file path and format (CSV or JSON).
    Args:
        file_path (str): The path to the file containing articles.
        file_format (str): The format of the file ('csv' or 'json').
        num (int, optional): The maximum number of articles to load. If None, all articles are loaded.
    Returns:
        list: A list of articles as dictionaries.
    """
    articles = []
    if file_format=="csv":
        with open(file_path, encoding="utf-8") as csvfile:
            reader = DictReader(csvfile)
            for row in reader:
                articles.append(row)
    elif file_format=="json":
        with open(file_path, encoding="utf-8") as jsonfile:
            articles = json.loads(jsonfile.read())
    for row in articles:
        if row["title"] is None:
            row["title"] = row["text"][:30]
    if num:
        shuffle(articles)
        articles = articles[:num]
    print(len(articles),"articles loaded")
    return articles


def init_recommendations(n, articles):
    """
    Initialize a list of random article recommendations.
    Args:
        n (int): The number of recommendations to generate.
        articles (list): The list of articles from which to recommend.
    Returns:
        list: A list of indices corresponding to recommended articles.
    """
    return sample(range(len(articles)), n)


def display_article_titles(recommendations, articles):
    """
    Display the titles of articles based on the provided recommendations.
    Args:
        recommendations (list): List of indices of recommended articles.
        articles (list): The list of articles containing the titles.
    """
    for i in range(len(recommendations)):
        art_num = recommendations[i]
        print(f'{i + 1}. {articles[art_num]["title"]}')


def display_recommendations(recommendations, articles, isInit):
    """
    Display a list of recommended articles based on whether this is the initial display.
    Args:
        recommendations (list): List of indices of recommended articles.
        articles (list): The list of articles containing the titles.
        isInit (bool): Indicates if this is the initial display of recommendations.
    """
    if isInit:
        print("\n\n\nHere is a list of articles for you:\n")
        display_article_titles(recommendations, articles)
    else:
        print("\n\n\nHere are some new recommendations for you:\n")
        display_article_titles(recommendations[:MAX_NEW_RECS], articles)

        print("\nOr if you want something different, how about...\n")
        for i in range(len(recommendations) - 2, len(recommendations)):
            art_num = recommendations[i]
            print(f'{i + 1}. {articles[art_num]['title']}')


def display_article(art_num, articles):
    """
    Display the full content of a specific article.
    Args:
        art_num (int): The index of the article to display.
        articles (list): The list of articles containing the content.
    """
    print("\n\n")
    print("article",art_num)
    print("=========================================")
    print(articles[art_num]["title"])
    print()
    print(articles[art_num]["text"])
    print("=========================================")
    print("\n\n")


def new_recommendations(last_choice, n, articles, vectors):
    """
    Generate new article recommendations based on the last chosen article.
    Args:
        last_choice (int): The index of the last chosen article.
        n (int): The number of recommendations to generate.
        articles (list): The list of articles.
        vectors: The vector representations of the articles for similarity computation.
    Returns:
        list: A list of indices corresponding to the new recommended articles.
    """
    last_article_vector = vectors[last_choice].reshape(1, -1)
    similarities = cosine_similarity(last_article_vector, vectors).flatten()
    similar_articles = similarities.argsort()[::-1]

    similar_articles = similar_articles[similar_articles != last_choice]
    top_similar = similar_articles[:8]
    dissimilar_articles = similar_articles[-2:]

    dissimilar_articles = dissimilar_articles.tolist()
    shuffle(dissimilar_articles)
    
    recommendations = list(top_similar) + list(dissimilar_articles)

    return recommendations


def choose_file():
    """
    Prompt the user to choose a file to load articles from.
    Returns:
        str: The path to the chosen file.
    """
    print("\nWelcome to the article database recommender system. choose a file to explore:\n")
    print("1. wikipedia_sample.json")
    print("2. arxiv_abstracts.csv")
    print("3. bbc_news.csv")
    print("4. Quit")
    while True:
        try:
            choice = int(input("\nYour choice? "))
        except ValueError:
            print("Invalid choice. Please enter a valid number.")
            continue

        if choice == 1:
            return "data/wikipedia_sample.json"
        elif choice == 2:
            return "data/arxiv_abstracts.csv"
        elif choice == 3:
            return "data/bbc_news.csv"
        elif choice == 4:
            sys.exit()
        else:
            print("Invalid choice. Please choose a number between 1 and 3. Or enter 4 to quit")


def get_file_type(file_path):
    """
    Determine the file type based on the file path.
    Args:
        file_path (str): The path to the file.
    Returns:
        str: The format of the file ('json' or 'csv').
    """
    return "json" if file_path == "data/wikipedia_sample.json" else "csv"          


def get_vectors(articles):
    """
    Generate vector representations for the articles using CountVectorizer.
    Args:
        articles (list): The list of articles to vectorize.
    Returns:
        Sparse matrix: The vector representations of the articles.
    """
    vectorizer = CountVectorizer(max_df=0.9, min_df=3)
    docs = []
    docnames = []
    for article_num in range(len(articles)):
        try:
            docs.append(articles[article_num]['text'])
            docnames.append(articles[article_num]['title'])
        except:
            pass
    
    vectors = vectorizer.fit_transform(docs)
    return vectors


def main():
    """
    Main function that runs the article recommender system.
    It handles file selection, article loading, and user interaction for displaying
    recommendations and articles.
    """
    global isInit
    while True:
        file = choose_file()
        file_type = get_file_type(file)
        articles = load_articles(file, file_type)
        vectors = get_vectors(articles)

        print("\n\n")
        recs = init_recommendations(NUM_RECS, articles)

        while True:
            display_recommendations(recs, articles, isInit)
            print("\nEnter 'q' to quit the application or 'r' to return to the file selection menu.\n")
            choice = input("\nYour choice? ")

            if choice.lower() == 'q':  
                print("Goodbye!")
                return
            elif choice.lower() == 'r':  
                isInit = True 
                break  

            try:
                choice = int(choice) - 1
            except ValueError:
                print("Invalid choice. Please enter a valid number or 'r' to return or 'q' to quit.")
                continue

            if choice < 0 or choice >= len(recs):
                print("Invalid Choice. Goodbye!")
                break

            display_article(recs[choice], articles)
            input("Press Enter")
            recs = new_recommendations(recs[choice], NUM_RECS, articles, vectors)
            isInit = False

if __name__ == "__main__":
    main()