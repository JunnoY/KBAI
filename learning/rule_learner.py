import itertools
import json
import math
import re
from contextlib import contextmanager
from dataclasses import dataclass, field, astuple
from typing import List, Generator, Union

from pyswip import Prolog

from learning.util import Algorithm, Examples, Dataset, Example, AlgorithmRegistry

from logging import getLogger

logger = getLogger(__name__)


@dataclass(frozen=True)
class Predicate:
    """
    Object representation of a predicate. Contains `name` which is the name of the predicate and its `arity`.
    """
    name: str
    arity: int

    def __post_init__(self):
        assert self.name[0].islower()


@dataclass(frozen=True)
class Expression:
    """
    Abstract base class representing a valid logical statement.
    """
    ...


@dataclass(frozen=True)
class Literal(Expression):
    """
    Literal: A Predicate with instantiated values for its arguments, which can be either variables or atomic values.

    Converting the literal to string will yield its syntactically valid prolog representation.
    """
    predicate: Predicate = field(hash=True)
    arguments: List[Union['Expression', str]] = field(hash=True)

    def __post_init__(self):
        """
        Make sure that the number of arguments corresponds to the predicate's arity.

        """
        assert (len(self.arguments) == self.predicate.arity,
                f"Number of arguments {len(self.arguments)} not "
                f"equal to the arity of the predicate {self.predicate.arity}")

    def __repr__(self):
        """
        Prolog representation.

        Returns: A syntactically valid prolog representation of the literal.

        """
        return f"{self.predicate.name}({','.join(str(a) for a in self.arguments)})"

    @classmethod
    def from_str(cls, string):
        """
        Generates a python object from a syntactically valid prolog representation.
        Args:
            string: Prolog representation of the literal.

        Returns: `Literal` object equivalent to the prolog representation.

        """
        predicate = get_predicate(string)
        args = get_args(string)
        return Literal(predicate, args)


def get_predicate(text: str) -> Predicate:
    """
    Returns the name and arity of a predicate from a syntactically valid prolog representation.
    Args:
        text: Text to extract the predicate from.

    Returns: Object of `Predicate` class with its corresponding name and arity.

    """
    text = str(text)
    name = text[:text.find("(")].strip()
    arity = len(re.findall("Variable", text))
    if arity == 0:
        arity = len(re.findall(",", text)) + 1
    return Predicate(name, arity)


@dataclass(frozen=True)
class Disjunction(Expression):
    """
    Represents a disjunction of horn clauses which is initially empty.
    """
    expressions: List['HornClause'] = field(default_factory=list)

    def generalize(self, expression: 'HornClause'):
        """
        Adds another horn clause to the disjunction.
        Args:
            expression: Horn clause to add

        """
        self.expressions.append(expression)

    def __repr__(self):
        """
        Returns a syntactically valid prolog representation of the horn clauses.

        Since there is no real disjunction in prolog, this is just a set of the expressions as separate statements.
        Returns:
            syntactically valid prolog representation of the contained horn clauses.

        """
        return " .\n".join(repr(e) for e in self.expressions) + ' .'


@dataclass(frozen=True)
class Conjunction(Expression):
    """
    Represents a conjunction of literals which is initially empty.
    """
    expressions: List[Expression] = field(default_factory=list)

    def specialize(self, expression: Expression):
        """
        Adds another literal to the conjunction.
        Args:
            expression: literal to add

        """
        self.expressions.append(expression)

    def __repr__(self):
        """
        Returns a syntactically valid prolog representation of the conjunction of the literals.

        Returns:
            syntactically valid prolog representation of the conjunction (comma-separated).

        """
        return " , ".join(repr(e) for e in self.expressions)


@dataclass(frozen=True)
class HornClause(Expression):
    """
    Represents a horn clause with a literal as `head` and a conjunction as `body`.
    """
    head: Expression
    body: Conjunction = field(default_factory=lambda: Conjunction())

    def get_vars(self):
        """
        Returns all variables appearing in the horn clause.

        Returns: All variables in the horn clause, according to prolog syntax, where variables are capitalised.

        """
        return re.findall(r"(?:[^\w])([A-Z]\w*)", str(self))

    def __repr__(self):
        """
        Converts to a syntactically valid prolog representation.

        Returns:
            Syntactically valid prolog representation of a horn clause in the form of
            ``head :- literal_1 , literal_2 , ... , literal_n``
            for all literals in the body.
        """
        return f"{str(self.head)} :- {' , '.join(str(e) for e in self.body.expressions)}"


def get_args(text: str) -> List[str]:
    """
    Returns the arguments of a text that is assumed to be a single literal in prolog representation.

    Args:
        text: Text to extract the arguments from. Must be valid prolog representation of a single literal.

    Returns:
        All arguments that appear in that literal.

    """
    return [x.strip() for x in re.findall(r"\(.*\)", str(text))[0][1:-1].split(",")]


@AlgorithmRegistry.register('foil')
@contextmanager
def FOIL(dataset: Dataset, recursive=False):
    f = _FOIL(dataset, recursive)
    try:
        yield f
    finally:
        f.abolish()


class _FOIL(Algorithm):
    prolog: Prolog

    def __init__(self, dataset: Dataset, recursive=False):
        super().__init__(dataset)
        logger.info("Creating prolog...")
        self.prolog = Prolog()

        self.recursive = recursive

        self.unique_var_count = 0

        if dataset.kb:
            logger.debug(f"Consulting {self.dataset.kb}")
            self.prolog.consult(self.dataset.kb)

    def abolish(self):
        for p, a in (astuple(a) for a in self.get_predicates()):
            self.prolog.query(f"abolish({p}/{a})")

    def predict(self, example: Example) -> bool:
        return any(self.covers(clause=c, example=example) for c in self.hypothesis.expressions)

    def get_predicates(self) -> List[Predicate]:
        """
        This method returns all (relevant) predicates from the knowledge base.

        Returns:
            all currently known predicates in the knowledge base that was loaded from the file corresponding to the
            dataset.

        """
        # Use "predicate_property(X,Y)" to get all the possible predicates in this documents, use
        # kb_path = self.dataset.kb to extract the predicates in the desired kb_path
        # append the desired predicates in the predicate_list and return it
        predicate_list = []
        kb_path = self.dataset.kb
        for i in self.prolog.query("predicate_property(X,Y)"):
            if kb_path in i['Y']:
                predicate_list.append(get_predicate(i['X']))
        return predicate_list

        # raise NotImplementedError()

    def find_hypothesis(self) -> Disjunction:
        """
        Initiates the FOIL algorithm and returns the final disjunction from the list that is returned by
        `FOIL.foil`.

        Returns: Disjunction of horn clauses that represent the learned target relation.

        """
        positive_examples = self.dataset.positive_examples
        negative_examples = self.dataset.negative_examples
        target = Literal.from_str(self.dataset.target)
        predicates = self.get_predicates()

        assert predicates

        clauses = self.foil(positive_examples, negative_examples, predicates, target)
        return Disjunction(clauses)

    def foil(self, positive_examples: Examples, negative_examples: Examples, predicates: List[Predicate],
             target: Literal) -> List[HornClause]:
        """
        Learns a list of horn clauses from a set of positive and negative examples which as a disjunction
        represent the hypothesis inferred from the dataset.

        This method is the outer loop of the foil algorithm.

        Args:
            positive_examples: Positive examples for the target relation to be learned.
            negative_examples: Negative examples for the target relation to be learned.
            predicates: Predicates that are allowed in the bodies of the horn clauses.
            target: Signature of the target relation to be learned

        Returns:
            A list of horn clauses that as a disjunction cover all positive and none of the negative examples.

        """
        # return [] if predicates is empty
        if len(predicates) == 0 or predicates == None:
            return []
        clauses = []
        # Get all the possible variables for the certain target
        variables_in_target = HornClause(target).get_vars()
        while len(positive_examples) != 0:
            clause = self.new_clause(positive_examples, negative_examples, predicates, target)
            # get all the positive examples from the covered clauses
            postive_covered = [e for e in positive_examples if self.covers(clause, e)]
            # get all the positive examples from the non-covered clauses
            positive_examples = [e for e in positive_examples if not self.covers(clause, e)]
            # for the recursive case, get all the positive examples from the covered clauses, for all variables in the
            # replace the variable in the target to the item in pos of positive_covered
            if self.recursive:
                for pos in postive_covered:
                    x = target.__repr__()
                    for var in variables_in_target:
                        x = x.replace(var, pos[var])
                    # append the target queries to the knowledge base
                    self.prolog.assertz(str(x))
                    # append the target predicates if they are not in predicates
                if target.predicate not in predicates:
                    predicates.append(target.predicate)
            clauses.append(clause)
        return clauses

        # raise NotImplementedError()

    def covers(self, clause: HornClause, example: Example) -> bool:
        """
        This method checks whether an example is covered by a given horn clause under the current knowledge base.
        Args:
            clause: The clause to check whether it covers the examples.
            example: The examples to check whether it is covered by the clause.

        Returns:
            True if covered, False otherwise

        """
        # A function that check if a set is a subset of another set
        def is_dict_subset(subset, superset):
            return all(item in superset.items() for item in subset.items())
        # find the horn clauses
        horn_clauses = clause.body.expressions
        query_str = ""
        # form the query string according to the horn clauses
        for i in range(len(horn_clauses)):
            if i < len(horn_clauses) - 1:
                query_str = query_str + str(horn_clauses[i]) + ","
            else:
                query_str = query_str + str(horn_clauses[i])
        # query the knowledge base
        query = self.prolog.query(str(query_str))
        # as long as there is one query result which is a subset of example, return True, else return False
        for i in query:
            if is_dict_subset(example, i):
                return True
        return False

        # raise NotImplementedError()

    def new_clause(self, positive_examples: Examples, negative_examples: Examples, predicates: List[Predicate],
                   target: Literal) -> HornClause:
        """
        This method generates a new horn clause from a dataset of positive and negative examples, a target and a
        list of allowed predicates to be used in the horn clause body.

        This corresponds to the inner loop of the foil algorithm.

        Args:
            positive_examples: Positive examples of the dataset to learn the clause from.
            negative_examples: Negative examples of the dataset to learn the clause from.
            predicates: List of predicates that can be used in the clause body.
            target: Head of the clause to learn.

        Returns:
            A horn clause that covers some part of the positive examples and does not contradict any of the
            negative examples.

        """
        # Create a new horn clause
        clause = HornClause(target, Conjunction([]))
        while len(negative_examples) != 0:
            # generate new candidates
            candidates = self.generate_candidates(clause, predicates)
            c_list = []
            lit_list = []
            for c in candidates:
                c_list.append(c)
                # calculate the information gain for each candidate
                lit_list.append(self.foil_information_gain(c, positive_examples, negative_examples))
            if len(lit_list) > 0:
                # select the candidate with the maximum information gain as the literal
                max_arg = lit_list.index(max(lit_list))
                lit = c_list[max_arg]
                clause.body.expressions.append(lit)
                pos_ex_new = []
                neg_ex_new = []
                # add new positive examples to positive_examples with the new literal
                for pos in positive_examples:
                    for i in self.extend_example(pos, lit):
                        pos_ex_new.append(i)
                # add new negative examples to negative_examples with the new literal
                for neg in negative_examples:
                    for j in self.extend_example(neg, lit):
                        neg_ex_new.append(j)
                positive_examples = pos_ex_new
                negative_examples = neg_ex_new
        return clause

        # raise NotImplementedError()

    def get_next_literal(self, candidates: List[Expression], pos_ex: Examples, neg_ex: Examples) -> Expression:
        """
        Returns the next literal with the highest information gain as computed from a given dataset of positive and
        negative examples.
        Args:
            candidates: Candidates to choose the one with the highest information gain from.
            pos_ex: Positive examples of the dataset to infer the information gain from.
            neg_ex: Negative examples of the dataset to infer the information gain from.

        Returns:
            the next literal with the highest information gain as computed
            from a given dataset of positive and negative examples.

        """

    def foil_information_gain(self, candidate: Expression, pos_ex: Examples, neg_ex: Examples) -> float:
        """
        This method calculates the information gain (as presented in the lecture) of an expression according
           to given positive and negative examples observations.

        Args:
               candidate: Attribute to infer the information gain for.
               pos_ex: Positive examples to infer the information gain from.
               neg_ex: Negative examples to infer the information gain from.

        Returns: The information gain of the given attribute according to the given observations.

        """
        # create a list of positive examples and negative examples, respectively
        pos_ex_new = []
        neg_ex_new = []
        # extend the current positive examples with the candidate and append them to the new lists of
        # positive examples
        for pos in pos_ex:
            for i in self.extend_example(pos, candidate):
                pos_ex_new.append(i)
        #  extend the current negative examples with the candidate and append them to the new lists of
        #  negative examples
        for neg in neg_ex:
            for j in self.extend_example(neg, candidate):
                neg_ex_new.append(j)
        # calculate the information gain
        p_1, n_1, p_0, n_0 = len(pos_ex_new), len(neg_ex_new), len(pos_ex), len(neg_ex)
        if p_1 != 0 and p_0 != 0:
            t = 0
            for p in pos_ex:
                if is_represented_by(p, pos_ex_new):
                    t += 1
            return t * (math.log2(p_1 / (p_1 + n_1)) - math.log2(p_0 / (p_0 + n_0)))
        return 0
        # raise NotImplementedError()

    def generate_candidates(self, clause: HornClause, predicates: List[Predicate]) -> Generator[Expression, None, None]:
        """
        This method generates all reasonable (as discussed in the lecture) specialisations of a horn clause
        given a list of allowed predicates.

        Args:
            clause: The clause to calculate possible specialisations for.
            predicates: Allowed predicate vocabulary to specialise the clause.

        Returns:
            All expressions that could be a reasonable specialisation of `clause`.

        """

        # find all permutations of variables that follows the rules
        def permutations(values, num, restricted, check_list):
            result = []

            def backtrack(values_to_consider, curr_permutation, remaining, used):
                if remaining == 0:
                    # Check that at least one value in the permutation is in the check_list
                    for val in curr_permutation:
                        if val in check_list:
                            result.append(curr_permutation)
                            break
                else:
                    for i in range(len(values_to_consider)):
                        val = values_to_consider[i]
                        if val in restricted and val in used:
                            continue
                        curr_permutation.append(val)
                        if val in restricted:
                            used.add(val)
                        backtrack(values_to_consider, curr_permutation[:], remaining - 1, used)
                        curr_permutation.pop()
                        if val in restricted:
                            used.remove(val)

            backtrack(values, [], num, set())
            return result

        # find all names of clauses, and combine each names with all pairs of variables
        # return the new candidates
        index1 = str(clause.head).find("(")
        index2 = str(clause.head).find(")")
        if len(predicates) > 0:
            for predicate in predicates:
                self.unique_var_count = 0
                name = predicate.name
                arity = predicate.arity
                unique_var_list = []
                variables = str(clause.head)[index1 + 1:index2].split(",")
                kb_list = variables.copy()
                if arity > 1:
                    for i in range(arity - 1):
                        unique_var = self.unique_var()
                        unique_var_list.append(unique_var)
                        variables.append(unique_var)
                possible_combinations = permutations(variables, arity, unique_var_list, kb_list)
                for comb in possible_combinations:
                    expression = name + "("
                    for i in range(len(comb)):
                        if i != len(comb) - 1:
                            expression = expression + comb[i] + ","
                        else:
                            expression = expression + comb[i] + ")"
                    yield Literal.from_str(expression)
        # raise NotImplementedError()

    def extend_example(self, example: Example, new_expr: Expression) -> Generator[Example, None, None]:
        """
        This method extends an example with all possible substitutions subject to a given expression and the current
        knowledge base.
        Args:
            example: Example to extend.
            new_expr: Expression to extend the example with.

        Returns:
            A generator that yields all possible substitutions for a given example an an expression.

        """
        # form a new prolog
        current_prolog = self.prolog
        current_prolog.consult(self.dataset.kb)
        index1 = str(new_expr).find("(")
        index2 = str(new_expr).find(")")
        # extract the names of examples
        list1 = list(example.keys())
        # extract the names of new expressions
        list2 = str(new_expr)[index1 + 1:index2].split(",")
        # find common names in list 1 and list 2
        common_keys = set(list1).intersection(list2)
        # query the new expression and generate the extended example the common names
        for i in current_prolog.query(str(new_expr)):
            valid = True
            for key in common_keys:
                if i[key] != example[key]:
                    valid = False
            if valid:
                for key in common_keys:
                    i.pop(key)
                example_copy = example.copy()
                example_copy.update(i)
                yield example_copy

        # raise NotImplementedError()

    def unique_var(self) -> str:
        """
        Returns the next uniquely numbered variable to be used.

        Returns:
            the next uniquely named variable in the following format: `V_i` where `i` is a number.

        """
        # get how many unique variables currently, generate a new variable with the next unique number
        variable = "V_" + str(self.unique_var_count)
        self.unique_var_count += 1
        return variable
        # raise NotImplementedError()


def is_represented_by(example: Example, examples: Examples) -> bool:
    """
    Checks whether a given example is represented by a list of examples.
    Args:
        example: Example to check whether it's represented.
        examples: Examples to check whether they represent the example.

    Returns:
        True, if for some `e` in `examples` for all variables (keys except target) in `example`,
        the values are equal (potential additional variables in `e` do not need to be considered). False otherwise.

    """
    valid_set = []
    for q in examples:
        valid = True
        for key in example.keys():
            if key != 'target':
                if example[key] != q[key]:
                    valid = False
        valid_set.append(valid)
    return True in valid_set
    # raise NotImplementedError()

