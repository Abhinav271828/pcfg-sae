from typing import Iterator, List, Tuple, Union
import random
import numpy as np
import nltk  # type: ignore
from nltk.grammar import ProbabilisticProduction  # type: ignore
from nltk.grammar import Nonterminal  # type: ignore
from .utils import define_prior

Symbol = Union[str, Nonterminal]


class ProbabilisticGenerator(nltk.grammar.PCFG):
    def generate(self, n: int = 1) -> Iterator[str]:
        """Probabilistically, recursively reduce the start symbol `n` times,
        yielding a valid sentence each time.

        Args:
            n: The number of sentences to generate.

        Yields:
            The next generated sentence.
        """
        for _ in range(n):
            x = self._generate_derivation(self.start())
            yield x

    def _generate_derivation(self, nonterminal: Nonterminal) -> str:
        """Probabilistically, recursively reduce `nonterminal` to generate a
        derivation of `nonterminal`.

        Args:
            nonterminal: The non-terminal nonterminal to reduce.

        Returns:
            The derived sentence.
        """
        sentence: List[str] = []
        symbol: Symbol
        derivation: str

        for symbol in self._reduce_once(nonterminal):
            if isinstance(symbol, str):
                derivation = symbol
            else:
                derivation = self._generate_derivation(symbol)

            if derivation != "":
                sentence.append(derivation)

        return " ".join(sentence)

    def _reduce_once(self, nonterminal: Nonterminal) -> Tuple[Symbol]:
        """Probabilistically choose a production to reduce `nonterminal`, then
        return the right-hand side.

        Args:
            nonterminal: The non-terminal symbol to derive.

        Returns:
            The right-hand side of the chosen production.
        """
        return self._choose_production_reducing(nonterminal).rhs()

    def _choose_production_reducing(
        self, nonterminal: Nonterminal
    ) -> ProbabilisticProduction:
        """Probabilistically choose a production that reduces `nonterminal`.

        Args:
            nonterminal: The non-terminal symbol for which to choose a production.

        Returns:
            The chosen production.
        """
        productions: List[ProbabilisticProduction] = self._lhs_index[nonterminal]
        probabilities: List[float] = [production.prob() for production in productions]
        return random.choices(productions, weights=probabilities)[0]



class PCFG:

    def __init__(
            self,
            n_nouns: int = 10,
            n_verbs: int = 10,
            n_adjectives: int = 10,
            n_pronouns: int = 10,
            n_adverbs: int = 10,
            n_conjunctions: int = 2,
            alpha: float = 1e5,
            prior_type: str = 'dirichlet',
            tasks: dict = None,
            seed: int = 42,
            ):
        """Define the PCFG object.

        Args:
            n_nouns: The number of nouns in the vocabulary.
            n_verbs: The number of verbs in the vocabulary.
            n_adjectives: The number of adjectives in the vocabulary.
            n_pronouns: The number of pronouns in the vocabulary.
            n_adverbs: The number of adverbs in the vocabulary.
            n_conjunctions: The number of conjunctions in the vocabulary.
            alpha: The concentration parameter for the Dirichlet distribution.
            prior_type: The type of prior distribution.
            tasks: The tasks to perform.
            seed: The random seed.

        Returns:
            PCFG: A PCFG object.
        """

        # Set the random seed
        random.seed(seed)
        np.random.seed(seed)

        # Concept classes object
        self.n_nouns = n_nouns
        self.n_verbs = n_verbs
        self.n_adjectives = n_adjectives
        self.n_pronouns = n_pronouns
        self.n_adverbs = n_adverbs
        self.n_conjunctions = n_conjunctions
        self.alpha = alpha
        self.prior_type = prior_type
        
        # Grammar
        self.production_rules = None
        self.lexical_symbolic_rules = None
        self.grammar = self.create_grammar(
            n_nouns=n_nouns,
            n_verbs=n_verbs,
            n_adjectives=n_adjectives,
            n_pronouns=n_pronouns,
            n_adverbs=n_adverbs,
            n_conjunctions=n_conjunctions,
        )

        # Tasks
        self.tasks = tasks

        # Set the vocabulary
        self.vocab, self.id_to_token_map, self.vocab_size = self.gather_vocabulary()

        # Parser
        self.parser = nltk.ViterbiParser(self.grammar)


    def create_grammar(
            self, 
            n_nouns: int,
            n_verbs: int,
            n_adjectives: int,
            n_pronouns: int,
            n_adverbs: int,
            n_conjunctions: int,
            ):
        """Define the PCFG grammar.

        Args:
            n_nouns: The number of nouns in the vocabulary.
            n_verbs: The number of verbs in the vocabulary.
            n_adjectives: The number of adjectives in the vocabulary.
            n_pronouns: The number of pronouns in the vocabulary.
            n_adverbs: The number of adverbs in the vocabulary.
            n_conjunctions: The number of conjunctions in the vocabulary.

        Returns:
            The PCFG grammar.
        """

        # Define production rules
        self.production_rules = """
                S -> NP VP [1.0] | VP NP [0.0] 
                NP -> Adj N [0.5] | NP Conj NP [0.25] | Pro [0.25]
                VP -> V [0.25] | V NP [0.35] | VP Adv [0.25] | VP Conj VP [0.15] 
                """
        
        self.lexical_symbolic_rules = ""

        ## Define lexical rules
        symbol_types = ['N', 'V', 'Adj', 'Pro', 'Adv', 'Conj']
        n_symbol_to_tokens = [n_nouns, n_verbs, n_adjectives, n_pronouns, n_adverbs, n_conjunctions]
        token_prefix = ['noun', 'verb', 'adj', 'pro', 'adv', 'conj']

        for symbol_type, n_symbol_to_token, prefix in zip(symbol_types, n_symbol_to_tokens, token_prefix):
            prior_over_symbol = define_prior(n_symbol_to_token, alpha=self.alpha, prior_type=self.prior_type)
            rhs_symbol = ""
            for i in range(n_symbol_to_token):
                rhs_symbol += f"'{prefix}{i}' [{prior_over_symbol[i]}] | "
            rhs_symbol = rhs_symbol[:-3]
            self.lexical_symbolic_rules += f"{symbol_type} -> {rhs_symbol} \n"

        # Create the grammar
        return ProbabilisticGenerator.fromstring(self.production_rules + self.lexical_symbolic_rules)


    def gather_vocabulary(self):
        """Gather the vocabulary from the concept classes.

        Returns:
            The vocabulary.
        """

        # Gather concept classes' vocabulary
        n_symbol_to_tokens = [self.n_nouns, self.n_verbs, self.n_adjectives, self.n_pronouns, self.n_adverbs, self.n_conjunctions]
        token_prefix = ['noun', 'verb', 'adj', 'pro', 'adv', 'conj']
        vocab = {}
        vocab_size = 0
        for prefix, n_symbol_to_token in zip(token_prefix, n_symbol_to_tokens):
            for i in range(n_symbol_to_token):
                vocab[f'{prefix}{i}'] = vocab_size
                vocab_size += 1
        vocab_size = len(vocab)

        # Add special tokens to be used for defining sequences in dataloader
        for special_token in ['<pad>', 'Task:', '<null>', 'Ops:', 'Out:', '\n', '<eos>', '<sep>']:
            vocab[special_token] = vocab_size
            vocab_size += 1

        # Add task tokens
        for task_token in self.tasks:
            vocab[task_token] = vocab_size
            vocab_size += 1

        # Create an inverse vocabulary
        id_to_token_map = {v: k for k, v in vocab.items()}

        return vocab, id_to_token_map, vocab_size


    def tokenize_sentence(self, sentence: str) -> List[int]:
        """Tokenize a sentence.

        Args:
            sentence: The sentence to tokenize.

        Returns:
            The tokenized sentence.
        """

        # Tokenize the sentence
        tokens = sentence.split(' ')

        # Convert the tokens to indices
        token_indices = []
        for token in tokens:
            if token == '' or token == ' ':
                continue
            else:
                token_indices.append(self.vocab[token])

        return token_indices


    def detokenize_sentence(self, token_indices) -> str:
        """Detokenize a sentence.

        Args:
            token_indices: The token indices to detokenize.

        Returns:
            The detokenized sentence.
        """

        # Convert the indices to tokens
        tokens = [self.id_to_token_map[token] for token in np.array(token_indices)]

        # Detokenize the tokens
        sentence = " ".join(tokens)

        return sentence


    def sentence_generator(
            self, 
            num_of_samples: int,
            ) -> Iterator[str]:
        """
        1. Generate a sentence from the grammar
        2. Fill the sentence with values from the concept classes
        """

        # An iterator that dynamically generates symbolic sentences from the underlying PCFG
        symbolic_sentences = self.grammar.generate(num_of_samples)

        # Fill the sentences with values from the concept classes
        for s in symbolic_sentences:
            yield s



    def check_grammaticality(self, sentence: str) -> bool:
        """Check if a sentence is in the grammar.

        Args:
            sentence: The sentence to check.

        Returns:
            Whether the sentence is in the grammar.
        """

        # Remove instruction decorator and pad tokens
        if 'Out:' in sentence:
            sentence = sentence.split('Out: ')
            sentence = sentence[1] if len(sentence) > 1 else sentence[0]
        if '<pad>' in sentence:
            sentence = sentence.split(' <pad>')
            sentence = sentence[0] if len(sentence) > 1 else sentence[0]

        # Tokenize the sentence
        tokens = sentence.split(' ')
        if '' in tokens:
            tokens.remove('')

        # Run parser
        try:
            parser_output = self.parser.parse(tokens).__next__()
            logprobs, height = parser_output.logprob(), parser_output.height()
            return (True, logprobs, height, None), len(tokens)
        except:
            failure = ' '.join(tokens)
            return (False, None, None, failure), len(tokens)
