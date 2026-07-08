import arxiv


# -----------------------------------
# 1. Search query
# -----------------------------------
# For testing, we use the topic from your sample paper.
query = "Retrieval Augmented Generation scientific literature question answering PaperQA"


# -----------------------------------
# 2. Create arXiv search
# -----------------------------------
search = arxiv.Search(
    query=query,
    max_results=5,
    sort_by=arxiv.SortCriterion.Relevance
)


# -----------------------------------
# 3. Print related papers
# -----------------------------------
print("\n===== Related Papers from arXiv =====\n")

client = arxiv.Client()

for i, result in enumerate(client.results(search), start=1):
    print(f"Paper {i}")
    print("Title:", result.title)
    print("Authors:", ", ".join(author.name for author in result.authors))
    print("Published:", result.published.date())
    print("URL:", result.entry_id)
    print("\nAbstract:")
    print(result.summary[:1000])
    print("\n" + "-" * 80 + "\n")