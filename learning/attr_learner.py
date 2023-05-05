import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Dict

import numpy as np

from learning.util import Algorithm, AlgorithmRegistry


Example = Dict[str, Any]
Examples = List[Example]

from logging import getLogger

logger = getLogger(__name__)


@dataclass(frozen=True)
class AttrLogicExpression(ABC):
    """
    Abstract base class representing a logic expression.
    """
    ...

    @abstractmethod
    def __call__(self, *args, **kwargs):
        ...


@dataclass(frozen=True)
class Conjunction(AttrLogicExpression):
    """
    A configuration of attribute names and the values the attributes should take for this conjunction to evaluate
    to true.

    `attribute_confs` is a map from attribute names to their values.
    """
    attribute_confs: Dict[str, Any]

    def __post_init__(self):
        assert 'target' not in self.attribute_confs, "Nice try, but 'target' cannot be part of the hypothesis."

    def __call__(self, example: Example):
        """
        Evaluates whether the conjunction applies to an example or not. Returns true if it does, false otherwise.


        Args:
            example: Example to check if the conjunction applies.

        Returns:
            True if the values of *all* attributes mentioned in the conjunction and appearing in example are equal,
            false otherwise.


        """
        return all(self.attribute_confs[k] == example[k] for k in set(self.attribute_confs).intersection(example))

    def __repr__(self):
        return " AND ".join(f"{k} = {v}" for k, v in self.attribute_confs.items())


@dataclass(frozen=True)
class Disjunction(AttrLogicExpression):
    """
    Disjunction of conjunctions.
    """
    conjunctions: List[Conjunction]

    def __call__(self, example: Example):
        """
        Evaluates whether the disjunction applies to a given example.

        Args:
            example: Example to check if the disjunction applies.

        Returns: True if any of its conjunctions returns true, and false if none evaluates to true.

        """
        return any(c(example) for c in self.conjunctions)

    def __repr__(self):
        return " " + "\nOR\n ".join(f"{v}" for v in self.conjunctions)


class Tree(ABC):
    """
    This is an abstract base class representing a leaf or a node in a tree.
    """
    ...


@dataclass
class Leaf(Tree):
    """
    This is a leaf in the tree. It's value is the (binary) classification, either True or False.
    """
    target: bool


@dataclass
class Node(Tree):
    """
    This is a node in the tree. It contains the attribute `attr_name` which the node is splitting on and a dictionary
    `branches` that represents the children of the node and maps from attribute values to their corresponding subtrees.
    """
    attr_name: str
    branches: Dict[Any, Tree] = field(default_factory=dict)


def same_target(examples: Examples) -> bool:
    """
    This function checks whether the examples all have the same target.

    Args:
        examples: Observations to check

    Returns: Whether the examples all have the same target.
    """
    # Create an empty list called target_list, then add the target in all examples to the target list
    # Compare the remaining targets with the first target in the target_list, if all of them are the same, return true
    # else return false
    target_list = []
    for example in examples:
        target_list.append(example['target'])

    return all(x == target_list[0] for x in target_list)
    # raise NotImplementedError()


def plurality_value(examples: Examples) -> bool:
    """
    This function returns whether there are more positively or negatively classified examples in the dataset.
    Args:
        examples: Examples to check.

    Returns: True if more examples classified as positive, False otherwise.

    """
    # loop through all items in examples, if the target in item is True, then true_targets + 1, else false_targets + 1,
    # if true_targets > false_targets, return True, else return False
    true_targets = 0
    false_targets = 0
    for example in examples:
        if example['target'] is True:
            true_targets += 1
        else:
            false_targets += 1
    if true_targets > false_targets:
        return True
    else:
        return False
    # raise NotImplementedError()


def binary_entropy(examples: Examples) -> float:
    """
    Calculates the binary (shannon) entropy of a dataset regarding its classification.
    Args:
        examples: Dataset to calculate the shannon entropy.

    Returns: The shannon entropy of the classification of the dataset.

    """
    # Create a dictionary called class_counts
    # loop through all items in examples, if the target of the item is not in class_counts, add it to class counts and set
    # its count to 1
    # for an existed label, count += 1
    # find the entropy for all labels in class_counts and compute the final entropy
    class_counts = {}
    entropy = 0.0
    num_entries = len(examples)
    # Check if the dataset is empty
    if num_entries > 0:
        for example in examples:
            label = example['target']
            if label not in class_counts.keys():
                class_counts[label] = 1
            else:
                class_counts[label] += 1
        for key in class_counts.keys():
            prob = float((class_counts[key]) / num_entries)
            entropy -= prob * math.log2(prob)
    return entropy
    # raise NotImplementedError()


def to_logic_expression(tree: Tree) -> AttrLogicExpression:
    """
    Converts a Decision tree to its equivalent logic expression.
    Args:
        tree: Tree to convert.

    Returns: The corresponding logic expression consisting of attribute values, conjunctions and disjunctions.

    """
    """
    This code defines a function called "recursive_loop" that takes in two arguments: "tree", which is an instance of a Tree object, and "initial_name", which is a string representing the name of the node from which the function will start iterating.

    The function starts by checking whether the current node is the initial node. If it is, an empty dictionary is created for it. Then, the function iterates over each of the branches of the current node.
    
    For each branch, the function creates a temporary dictionary that is a copy of the dictionary associated with the current node. The function then adds the path from the current node to the temporary dictionary. 
    
    If the branch is a target (i.e., a leaf node with target=True), the function appends a Conjunction object to a list called "disjunction_paths". 
    
    Otherwise, the function updates the dictionary associated with the child node and recursively calls itself on the child node.
    
    Finally, the function returns the list of Conjunction objects created during the iteration.

    The two global variables, "conjunction_dict_list" and "disjunction_paths", are defined elsewhere in the code.
    """
    def recursive_loop(tree: Tree, initial_name):
        # if the current node is the initial node, initialise an emoty dict for it
        if tree.attr_name == initial_name:
            conjunction_dict_list[tree.attr_name] = {}
        for path in tree.branches.keys():
            current_node = tree.attr_name
            if tree.branches[path] != Leaf(target=False):
                temp_dict = conjunction_dict_list[current_node].copy()
                temp_dict[current_node] = path
                if tree.branches[path] == Leaf(target=True):
                    disjunction_paths.append(Conjunction(temp_dict))
                else:
                    # update dict; update the parent dict for the children of the current node
                    update_dict = conjunction_dict_list[current_node].copy()
                    update_dict[current_node] = path
                    conjunction_dict_list[tree.branches[path].attr_name] = update_dict.copy()
                    recursive_loop(tree.branches[path], initial_name)
        return disjunction_paths

    disjunction_paths = []
    conjunction_dict_list = {}
    initial_name = tree.attr_name
    expression = Disjunction(recursive_loop(tree, initial_name))
    return expression

    # raise NotImplementedError()


@AlgorithmRegistry.register("dtl")
class DecisionTreeLearner(Algorithm):
    """
    This is the decision tree learning algorithm.
    """

    def find_hypothesis(self) -> AttrLogicExpression:
        tree = self.decision_tree_learning(examples=self.dataset.examples, attributes=self.dataset.attributes,
                                           parent_examples=[])
        return to_logic_expression(tree)

    def decision_tree_learning(self, examples: Examples, attributes: List[str], parent_examples: Examples) -> Tree:
        """
        This is the main function that learns a decision tree given a list of example and attributes.
        Args:
            examples: The training dataset to induce the tree from.
            attributes: Attributes of the examples.
            parent_examples: Examples from previous step.

        Returns: A decision tree induced from the given dataset.
        """
        # return the Leaf with target = plurality_value of the parent examples if examples is empty
        if len(examples) == 0:
            return Leaf(target=plurality_value(parent_examples))
        # return the Leaf with target = first target of the first item in examples if all items in examples have the same target
        elif same_target(examples):
            return Leaf(target=examples[0]['target'])
        # return the Leaf with target = plurality_value of the examples if attributes is empty
        elif len(attributes) == 0:
            return Leaf(target=plurality_value(examples))
        # else generate the most important attributes, create a node with the name of the attribute and append subtrees
        # to this node
        else:
            attr = self.get_most_important_attribute(attributes, examples)
            tree = Node(attr_name=attr, branches={})
            new_attrs = set(attributes) - {attr}
            for example in examples:
                value = example[attr]
                new_examples = [e for e in examples if e[attr] == value]
                subtree = DecisionTreeLearner.decision_tree_learning(self, new_examples, list(new_attrs), examples)
                tree.branches[value] = subtree
            return tree

    def get_most_important_attribute(self, attributes: List[str], examples: Examples) -> str:
        """
        Returns the most important attribute according to the information gain measure.
        Args:
            attributes: The attributes to choose the most important attribute from.
            examples: Dataset from which the most important attribute is to be inferred.

        Returns: The most informative attribute according to the dataset.

        """
        # return "" if attributes is empty
        if len(attributes) == 0:
            return ""
        # set the highest_attribute to be the first item in attributes, set best_gain as the
        # information gain of the first item in attributes
        # compare the best one with the remaining attributes in the list, update the best attribute if the current
        # attribute has a higher information gain than the best one
        highest_attribute = attributes[0]
        best_gain = self.information_gain(examples, highest_attribute)
        for attribute in attributes:
            current_gain = self.information_gain(examples, attribute)
            if current_gain >= best_gain:
                highest_attribute = attribute
                best_gain = current_gain
        return highest_attribute
        # raise NotImplementedError()

    def information_gain(self, examples: Examples, attribute: str) -> float:
        """
        This method calculates the information gain (as presented in the lecture)
        of an attribute according to given observations.

        Args:
            examples: Dataset to infer the information gain from.
            attribute: Attribute to infer the information gain for.

        Returns: The information gain of the given attribute according to the given observations.

        """
        entropy = binary_entropy(examples)
        entropy_given_attrs = 0.0
        class_counts_attrs = {}
        class_counts_attrs_target = {}
        num_entries = len(examples)
        # Check if the dataset is empty
        if num_entries > 0:
            for example in examples:
                label = example[attribute]
                # get the label of the current example, and try to get its class_counts set from class_counts_attrs_target
                # if the label is not in class_counts_attrs_target, assign an empty set for it
                # if the label is in class_counts_attrs_target, assign it to its set in class_counts_attrs_target
                if label not in class_counts_attrs_target:
                    class_counts_attrs_target[label] = {}
                temp_dict = class_counts_attrs_target[label]
                # for each label, count how many is it in examples, append its count to its class_counts_attrs set
                if label not in class_counts_attrs.keys():
                    class_counts_attrs[label] = 1
                    if example['target'] not in temp_dict:
                        temp_dict[example['target']] = 1
                    else:
                        temp_dict[example['target']] += 1
                else:
                    class_counts_attrs[label] += 1
                    if example['target'] not in temp_dict:
                        temp_dict[example['target']] = 1
                    else:
                        temp_dict[example['target']] += 1
            # calculate the entropy for each label and compute overall entropy
            for key in class_counts_attrs.keys():
                prob1 = float((class_counts_attrs[key]) / num_entries)
                target_dict = class_counts_attrs_target[key]
                num_entries_target = sum(target_dict.values())
                temp_entropy = 0
                for value in target_dict.values():
                    temp_entropy -= (value / num_entries_target) * math.log2(value / num_entries_target)
                entropy_given_attrs += prob1 * temp_entropy

        ig = entropy - entropy_given_attrs
        return ig

        # raise NotImplementedError()


@AlgorithmRegistry.register("random-sampling-dtl")
class MyDecisionTreeLearner(DecisionTreeLearner):
    """
    This is the decision tree learning algorithm.
    """
    def get_most_important_attribute(self, attributes: List[str], examples: Examples) -> str:
        """
        Returns the most important attribute according to the information gain measure.
        Args:
            attributes: The attributes to choose the most important attribute from.
            examples: Dataset from which the most important attribute is to be inferred.

        Returns: The most informative attribute according to the dataset.

        """
        sample_attributes = random.sample(attributes, int(np.sqrt(len(attributes))))
        highest_attribute = sample_attributes[0]
        best_gain = self.information_gain(examples, highest_attribute)
        for attribute in sample_attributes:
            current_gain = self.information_gain(examples,attribute)
            if current_gain >= best_gain:
                highest_attribute = attribute
                best_gain = current_gain
        return highest_attribute
        # raise NotImplementedError()

@AlgorithmRegistry.register("same-sample-efficient-dtl")
class MyDecisionTreeLearner2(DecisionTreeLearner):
    """
    This is the decision tree learning algorithm.
    """
    def same_target(examples: Examples) -> bool:
        target_list = []
        for example in examples:
            target_list.append(example['target'])

        different_var = 0
        temp_tar = target_list[0]['target']
        for tar in target_list:
            if tar['target'] != temp_tar:
                different_var += 1
            if different_var/len(target_list) > 0.2:
                return False
        return True
