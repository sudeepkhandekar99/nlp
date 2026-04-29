'''
Detailed summary → RAG retrieves broad relevant chunks → LLM summarizes
Flashcards → RAG retrieves key chunks → LLM creates front/back cards
Quiz → RAG retrieves concepts → LLM creates difficulty-based questions
Knowledge graph → RAG retrieves concept-heavy chunks → LLM extracts entities and relationships
'''


from rag_layer import RAGQueryEngine, generate_answer


class LearningFeatures:
    def __init__(self):
        self.rag = RAGQueryEngine()

    def generate_detailed_summary(self):
        question = """
        Create a detailed structured summary of this content.
        Include:
        1. Main topic
        2. Key concepts
        3. Important definitions
        4. Step by step explanation
        5. Practical examples
        6. Things to remember
        """

        return self.rag.ask(question, top_k=6)

    def generate_flashcards(self, num_cards=5):
        question = f"""
        Create exactly {num_cards} flashcards from this content.

        Format:
        Flashcard 1
        Front: ...
        Back: ...

        Rules:
        - Use only the uploaded content
        - Keep front side short
        - Keep back side clear and useful
        """

        return self.rag.ask(question, top_k=6)

    def generate_quiz(self, difficulty="easy", num_questions=5):
        question = f"""
        Create exactly {num_questions} quiz questions from this content.

        Difficulty: {difficulty}

        Difficulty rules:
        - Easy: definition based and direct recall
        - Medium: conceptual understanding and small application
        - Hard: analytical, inference based, and deeper reasoning

        Format:
        Q1. ...
        A. ...
        B. ...
        C. ...
        D. ...
        Correct Answer: ...
        Explanation: ...

        Use only the uploaded content.
        """

        return self.rag.ask(question, top_k=6)

    def generate_knowledge_graph(self):
        question = """
        Extract a knowledge graph from this content.

        Return entities and relationships in this format:

        Entities:
        - Entity name: short description

        Relationships:
        - Entity A -> relationship -> Entity B

        Rules:
        - Use only the uploaded content
        - Focus on important concepts
        - Keep relationships clear and meaningful
        """

        return self.rag.ask(question, top_k=8)


def print_feature_result(title, result):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(result["answer"])

    print("\nRetrieved chunks used:")
    for chunk in result["retrieved_chunks"]:
        print(f"- Chunk {chunk['chunk_id']} | Score: {chunk['score']:.4f}")


if __name__ == "__main__":
    features = LearningFeatures()

    summary = features.generate_detailed_summary()
    print_feature_result("DETAILED SUMMARY", summary)

    flashcards = features.generate_flashcards(num_cards=5)
    print_feature_result("FLASHCARDS", flashcards)

    easy_quiz = features.generate_quiz(difficulty="easy", num_questions=5)
    print_feature_result("EASY QUIZ", easy_quiz)

    medium_quiz = features.generate_quiz(difficulty="medium", num_questions=5)
    print_feature_result("MEDIUM QUIZ", medium_quiz)

    hard_quiz = features.generate_quiz(difficulty="hard", num_questions=5)
    print_feature_result("HARD QUIZ", hard_quiz)

    graph = features.generate_knowledge_graph()
    print_feature_result("KNOWLEDGE GRAPH", graph)