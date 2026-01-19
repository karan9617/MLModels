import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import re


@dataclass
class Context:
    """Represents a piece of contextual knowledge"""
    id: str
    content: str
    category: str
    relevance_score: float
    timestamp: str
    usage_count: int = 0
    success_rate: float = 1.0


@dataclass
class Interaction:
    """Represents a user interaction"""
    query: str
    response: str
    contexts_used: List[str]
    feedback: Optional[str] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class ContextualMemory:
    """Manages contextual knowledge with relevance scoring"""

    def __init__(self, storage_path: str = "memory.json"):
        self.storage_path = storage_path
        self.contexts: Dict[str, Context] = {}
        self.load_memory()

    def add_context(self, content: str, category: str, relevance: float = 1.0):
        """Add new contextual knowledge"""
        ctx_id = f"{category}_{len(self.contexts)}_{datetime.now().timestamp()}"
        context = Context(
            id=ctx_id,
            content=content,
            category=category,
            relevance_score=relevance,
            timestamp=datetime.now().isoformat()
        )
        self.contexts[ctx_id] = context
        self.save_memory()
        return ctx_id

    def get_relevant_contexts(self, query: str, top_k: int = 5) -> List[Context]:
        """Retrieve most relevant contexts for a query"""
        scored_contexts = []
        query_lower = query.lower()
        query_words = set(re.findall(r'\w+', query_lower))

        for ctx in self.contexts.values():
            content_lower = ctx.content.lower()
            content_words = set(re.findall(r'\w+', content_lower))

            # Calculate relevance based on word overlap
            overlap = len(query_words & content_words)
            if overlap > 0:
                # Boost by existing relevance score and success rate
                score = overlap * ctx.relevance_score * ctx.success_rate
                scored_contexts.append((score, ctx))

        # Sort by score and return top_k
        scored_contexts.sort(reverse=True, key=lambda x: x[0])
        return [ctx for _, ctx in scored_contexts[:top_k]]

    def update_context_performance(self, ctx_id: str, success: bool):
        """Update context performance based on feedback"""
        if ctx_id in self.contexts:
            ctx = self.contexts[ctx_id]
            ctx.usage_count += 1
            # Update success rate with exponential moving average
            alpha = 0.3
            ctx.success_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * ctx.success_rate
            self.save_memory()

    def save_memory(self):
        """Persist memory to disk"""
        data = {ctx_id: asdict(ctx) for ctx_id, ctx in self.contexts.items()}
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load_memory(self):
        """Load memory from disk"""
        if os.path.exists(self.storage_path):
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.contexts = {
                    ctx_id: Context(**ctx_data)
                    for ctx_id, ctx_data in data.items()
                }


class CodeGenerator:
    """Generates code responses based on context"""

    def __init__(self):
        self.templates = {
            "python": {
                "function": "def {name}({params}):\n    \"\"\"{docstring}\"\"\"\n    {body}",
                "class": "class {name}:\n    \"\"\"{docstring}\"\"\"\n    \n    def __init__(self, {params}):\n        {body}",
            }
        }

    def generate_code(self, query: str, contexts: List[Context]) -> str:
        """Generate code based on query and contexts"""
        # Extract intent from query
        query_lower = query.lower()

        if "function" in query_lower or "def" in query_lower:
            return self._generate_function(query, contexts)
        elif "class" in query_lower:
            return self._generate_class(query, contexts)
        else:
            return self._generate_general_code(query, contexts)

    def _generate_function(self, query: str, contexts: List[Context]) -> str:
        """Generate a function based on query and contexts"""
        # Extract function name from query
        name_match = re.search(r'(?:function|def)\s+(\w+)', query.lower())
        name = name_match.group(1) if name_match else "generated_function"

        # Build function body from contexts
        body_lines = ["# Generated based on learned patterns"]
        for ctx in contexts:
            if "example" in ctx.category or "pattern" in ctx.category:
                body_lines.append(f"# Context: {ctx.content[:50]}...")

        body_lines.append("pass  # Implement logic here")

        return f"""def {name}(param1, param2):
    \"\"\"
    Generated function based on contextual knowledge.
    Query: {query}
    \"\"\"
{chr(10).join('    ' + line for line in body_lines)}
"""

    def _generate_class(self, query: str, contexts: List[Context]) -> str:
        """Generate a class based on query and contexts"""
        name_match = re.search(r'class\s+(\w+)', query.lower())
        name = name_match.group(1).title() if name_match else "GeneratedClass"

        return f"""class {name}:
    \"\"\"
    Generated class based on contextual knowledge.
    Query: {query}
    \"\"\"

    def __init__(self):
        # Initialize based on learned patterns
        pass
"""

    def _generate_general_code(self, query: str, contexts: List[Context]) -> str:
        """Generate general code snippet"""
        code_lines = ["# Generated code snippet"]
        code_lines.append(f"# Query: {query}\n")

        for ctx in contexts:
            code_lines.append(f"# Using context: {ctx.category}")

        code_lines.append("\n# Implementation")
        code_lines.append("pass  # Add your implementation")

        return "\n".join(code_lines)


class SelfImprovingAssistant:
    """Main assistant that learns and improves over time"""

    def __init__(self, memory_path: str = "assistant_memory.json"):
        self.memory = ContextualMemory(memory_path)
        self.code_generator = CodeGenerator()
        self.interactions: List[Interaction] = []
        self.learning_rate = 0.1

        # Initialize with some base knowledge
        self._initialize_base_knowledge()

    def _initialize_base_knowledge(self):
        """Bootstrap with fundamental programming knowledge"""
        base_knowledge = [
            ("Functions encapsulate reusable logic", "concept"),
            ("Classes model objects with state and behavior", "concept"),
            ("List comprehensions provide concise iteration", "pattern"),
            ("Exception handling improves robustness", "pattern"),
            ("Type hints improve code clarity", "best_practice"),
        ]

        for content, category in base_knowledge:
            if not any(ctx.content == content for ctx in self.memory.contexts.values()):
                self.memory.add_context(content, category, relevance=1.0)

    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a code generation query"""
        # Retrieve relevant contexts
        relevant_contexts = self.memory.get_relevant_contexts(query, top_k=5)

        # Generate code response
        code_response = self.code_generator.generate_code(query, relevant_contexts)

        # Create interaction record
        interaction = Interaction(
            query=query,
            response=code_response,
            contexts_used=[ctx.id for ctx in relevant_contexts]
        )
        self.interactions.append(interaction)

        return {
            "code": code_response,
            "contexts_used": len(relevant_contexts),
            "confidence": self._calculate_confidence(relevant_contexts),
            "interaction_index": len(self.interactions) - 1
        }

    def _calculate_confidence(self, contexts: List[Context]) -> float:
        """Calculate confidence based on context quality"""
        if not contexts:
            return 0.3

        avg_success = sum(ctx.success_rate for ctx in contexts) / len(contexts)
        avg_usage = sum(ctx.usage_count for ctx in contexts) / len(contexts)

        # Higher confidence with more successful, frequently used contexts
        confidence = 0.5 + (avg_success * 0.3) + (min(avg_usage / 10, 1.0) * 0.2)
        return min(confidence, 1.0)

    def provide_feedback(self, interaction_index: int, feedback: str, success: bool):
        """Learn from user feedback"""
        if interaction_index >= len(self.interactions):
            return

        interaction = self.interactions[interaction_index]
        interaction.feedback = feedback

        # Update context performance
        for ctx_id in interaction.contexts_used:
            self.memory.update_context_performance(ctx_id, success)

        # Extract and learn from feedback
        if success and feedback:
            self._learn_from_success(interaction, feedback)

        print(f"✓ Feedback recorded. Learning from {'successful' if success else 'unsuccessful'} interaction.")

    def _learn_from_success(self, interaction: Interaction, feedback: str):
        """Extract new knowledge from successful interactions"""
        # Create new context from successful pattern
        new_context = f"Query pattern: '{interaction.query[:50]}...' was successful"
        self.memory.add_context(new_context, "learned_pattern", relevance=1.0)

        # Extract keywords from feedback to improve future matching
        if "good" in feedback.lower() or "works" in feedback.lower():
            keywords = re.findall(r'\w+', interaction.query.lower())
            for keyword in keywords[:3]:  # Top 3 keywords
                self.memory.add_context(
                    f"Keyword '{keyword}' associated with successful outcome",
                    "keyword_pattern",
                    relevance=0.8
                )

    def expand_context(self, new_knowledge: str, category: str = "learned"):
        """Manually expand the assistant's knowledge base"""
        ctx_id = self.memory.add_context(new_knowledge, category, relevance=1.0)
        print(f"✓ Context expanded. New context ID: {ctx_id}")
        return ctx_id

    def get_statistics(self) -> Dict[str, Any]:
        """Get assistant performance statistics"""
        total_contexts = len(self.memory.contexts)
        total_interactions = len(self.interactions)

        categories = {}
        for ctx in self.memory.contexts.values():
            categories[ctx.category] = categories.get(ctx.category, 0) + 1

        avg_success = sum(
            ctx.success_rate for ctx in self.memory.contexts.values()
        ) / total_contexts if total_contexts > 0 else 0

        return {
            "total_contexts": total_contexts,
            "total_interactions": total_interactions,
            "categories": categories,
            "average_success_rate": avg_success,
            "memory_size_kb": os.path.getsize(self.memory.storage_path) / 1024
            if os.path.exists(self.memory.storage_path) else 0
        }


def interactive_mode():
    """Run the assistant in interactive mode"""
    assistant = SelfImprovingAssistant()

    print("=" * 60)
    print("  SELF-IMPROVING CODE ASSISTANT")
    print("=" * 60)
    print("\nCommands:")
    print("  • Type your code request (e.g., 'create a function to sort list')")
    print("  • Type 'stats' to see statistics")
    print("  • Type 'learn <knowledge>' to teach the assistant")
    print("  • Type 'quit' to exit\n")

    while True:
        try:
            # Get user query
            user_input = input("\n🤔 What code would you like me to generate? ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() == 'quit':
                print("\n👋 Goodbye! Your learning has been saved.")
                break

            elif user_input.lower() == 'stats':
                stats = assistant.get_statistics()
                print("\n📊 STATISTICS:")
                print(f"  Total contexts: {stats['total_contexts']}")
                print(f"  Total interactions: {stats['total_interactions']}")
                print(f"  Average success rate: {stats['average_success_rate']:.2%}")
                print(f"  Categories: {stats['categories']}")
                continue

            elif user_input.lower().startswith('learn '):
                knowledge = user_input[6:].strip()
                if knowledge:
                    assistant.expand_context(knowledge, "user_taught")
                else:
                    print("❌ Please provide knowledge to learn")
                continue

            # Process query
            print("\n⚙️  Generating code...")
            result = assistant.process_query(user_input)

            print(f"\n✨ Generated Code (Confidence: {result['confidence']:.0%}):")
            print("-" * 60)
            print(result['code'])
            print("-" * 60)

            # Ask for feedback
            feedback_input = input("\n💬 Was this helpful? (yes/no/comment): ").strip().lower()

            if feedback_input:
                if feedback_input in ['yes', 'y', 'good', 'great']:
                    assistant.provide_feedback(
                        result['interaction_index'],
                        "User approved",
                        success=True
                    )
                elif feedback_input in ['no', 'n', 'bad']:
                    assistant.provide_feedback(
                        result['interaction_index'],
                        "User rejected",
                        success=False
                    )
                else:
                    # Custom feedback
                    success = 'good' in feedback_input or 'yes' in feedback_input
                    assistant.provide_feedback(
                        result['interaction_index'],
                        feedback_input,
                        success=success
                    )

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


# Main entry point
if __name__ == "__main__":
    interactive_mode()