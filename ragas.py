from typing import List, Dict, Any
from datasets import Dataset
from ragas import evaluate as ragas_evaluate
from ragas.metrics import faithfulness, answer_relevance, context_recall, context_precision
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from core.config.settings import AppSettings

class RagasRunner:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        
        # Initialize local models via Ollama to run the Ragas evaluation locally
        self.eval_llm = Ollama(
            base_url=settings.llm.base_url,
            model=settings.llm.provider,
            temperature=0.0
        )
        self.eval_embeddings = OllamaEmbeddings(
            base_url=settings.llm.base_url,
            model=settings.embeddings.provider  # Configured to map your embedding variant
        )

    def evaluate_retrieval_pipeline(self, 
                                    queries: List[str], 
                                    retrieved_chunks: List[List[str]], 
                                    generated_reviews: List[str], 
                                    ground_truths: List[List[str]]) -> Dict[str, Any]:
        """
        Runs RAGAS evaluation across queries to monitor retrieval precision and generation faithfulness.
        """
        # Formulate dataset structure required natively by Ragas
        data = {
            "question": queries,
            "contexts": retrieved_chunks,
            "answer": generated_reviews,
            "ground_truth": ground_truths
        }
        
        dataset = Dataset.from_dict(data)
        
        # Execute the score matrix evaluation asynchronously/locally
        score_result = ragas_evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevance,
                context_recall,
                context_precision
            ],
            llm=self.eval_llm,
            embeddings=self.eval_embeddings
        )
        
        # Convert results to clean Python dictionary for API/Streamlit tracking ingestion
        return dict(score_result)