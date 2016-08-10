/*
 * Copyright 2014 Simone Filice and Giuseppe Castellucci and Danilo Croce and Roberto Basili
 * and Giovanni Da San Martino and Alessandro Moschitti
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package it.uniroma2.sag.kelp.examples.demo.pruning;

import it.uniroma2.sag.kelp.data.dataset.SimpleDataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.manipulator.TreeNodePruner;
import it.uniroma2.sag.kelp.data.representation.tree.TreeRepresentation;
import it.uniroma2.sag.kelp.data.representation.tree.node.nodePruner.PruneNodeIfLeaf;
import it.uniroma2.sag.kelp.data.representation.tree.node.nodePruner.PruneNodeLeafNumber;
import it.uniroma2.sag.kelp.data.representation.tree.node.nodePruner.PruneNodeNumberOfChildren;

/**
 * This class shows how to use Kelp to prune a tree dataset. In the
 * example, several kernels can be used to build your own classifier. <br>
 * <br>
 * 
 * A description of the tree representations can be found in [Croce et al(2011)]
 * and [Annesi et al(2014)]. <br>
 * <br>
 * 
 * @author Giovanni Da San Martino
 * 
 */
public class TreePruningDemo {

	public static void main(String[] args) {
		int totalNumberOfNodesBeforePruning;
		int totalNumberOfNodesAfterPruning;
		String representationTobePruned = "grct";
		TreeNodePruner pruner;
		try {
			/*
			 * Initializing the Log level
			 */
			System.setProperty("org.slf4j.simpleLogger.defaultLogLevel", "INFO");

			/*
			 * Read the training dataset
			 */
			SimpleDataset trainingSet = new SimpleDataset();
			trainingSet.populate("src/main/resources/qc/train_5500.coarse.klp.gz");
			
			/*
			 * Let's count the number of nodes of all trees of the representation
			 * grct in the dataset. 
			 */
			totalNumberOfNodesBeforePruning = computeTotalNumberOfNodes(trainingSet, 
					representationTobePruned);

			/*
			 * Creating a pruner object which limits the number of leaves to 5
			 */
			pruner = createMaxNumberOfLeavesPrunerObject(5, representationTobePruned);
			
			/*
			 * The pruner can describe itself, so that errors can be easily spotted
			 */
			System.out.format("About to apply the following pruning "
					+ "strategy:%n%s", pruner.describe());
			/*
			 * Applying pruning to the dataset
			 */
			trainingSet.manipulate(pruner);
			
			totalNumberOfNodesAfterPruning = computeTotalNumberOfNodes(trainingSet, 
					representationTobePruned);
			/*
			 * Printing statistics about the nodes removed
			 */
			System.out.format("Total number of nodes before / after = %d / %d "
					+ "(ratio of nodes pruned = %f)%n", totalNumberOfNodesBeforePruning, 
					totalNumberOfNodesAfterPruning, 
					(float) (totalNumberOfNodesBeforePruning-totalNumberOfNodesAfterPruning)/ (float) totalNumberOfNodesBeforePruning);

			/*
			 * Second Pruning Demo
			 */
			System.out.format("%nSecond Experiment%n%n");
			trainingSet = new SimpleDataset();
			trainingSet.populate("src/main/resources/qc/train_5500.coarse.klp.gz");
			totalNumberOfNodesBeforePruning = computeTotalNumberOfNodes(trainingSet, 
					representationTobePruned);
			/*
			 * This time we want to reduce the number of children of the root node only
			 */
			pruner = createMaxChildrenPrunerObject(3, "grct");
			System.out.format("About to apply the following pruning "
					+ "strategy:%n%s", pruner.describe());
			trainingSet.manipulate(pruner);
			totalNumberOfNodesAfterPruning = computeTotalNumberOfNodes(trainingSet, 
					representationTobePruned);
			System.out.format("Total number of nodes before / after = %d / %d "
					+ "(ratio of nodes pruned = %f)%n", totalNumberOfNodesBeforePruning, 
					totalNumberOfNodesAfterPruning, 
					(float) (totalNumberOfNodesBeforePruning-totalNumberOfNodesAfterPruning)/ (float) totalNumberOfNodesBeforePruning);

		} catch (Exception e1) {
			e1.printStackTrace();
		}
	}

	public static int computeTotalNumberOfNodes(SimpleDataset dataset, String representation) {
		int totNodes = 0;
		for(Example ex: dataset.getExamples()) {
			totNodes += ((TreeRepresentation) ex.getRepresentation(representation)).getNumberOfNodes();
		}
		return totNodes;
	}

	public static TreeNodePruner createMaxNumberOfLeavesPrunerObject(int maxLeaves, 
			String representation) {
		System.out.format("Creating a pruning object for reducing the number of "
				+ "leaves to a maximum of %d%n", maxLeaves);
		/*
		 * instantiating the class which checks, for every node, whether they 
		 * are supposed to be removed  
		 */
		PruneNodeLeafNumber leafNumberPruner = 
				new PruneNodeLeafNumber(maxLeaves);
		/*
		 * We instantiate a further pruning object for non-leaf nodes, which 
		 * marks as removable those nodes which are left with no children (after
		 * applying the pruning above). As a consequence, not only the leaves in
		 * excess are removed, but also most of their ascendant nodes.  
		 */
		PruneNodeIfLeaf internalNodePruner = new PruneNodeIfLeaf();
		/*
		 * Aside from passing to the constructor the two objects above, we pass 
		 * also the name representation the tree is associated with, and we 
		 * specify the visit of the tree is not limited.  
		 */
		TreeNodePruner maxNumberOfLeavesPrunerClass = new TreeNodePruner(leafNumberPruner, 
				representation, internalNodePruner, null, TreeNodePruner.UNLIMITED_RECURSION);
		return maxNumberOfLeavesPrunerClass;
	}
	
	public static TreeNodePruner createMaxChildrenPrunerObject(int maxChildren, 
			String representation) {
		System.out.format("Creating a pruning object for reducing the number of "
				+ "children of the root node to a maximum of %d%n", maxChildren);
		/*
		 * instantiating the class which checks, for every node, whether they 
		 * are supposed to be removed  
		 */
		PruneNodeNumberOfChildren maxChildrenPruner = 
				new PruneNodeNumberOfChildren(maxChildren); 
		/*
		 * Aside from passing to the constructor the object above, we pass 
		 * also the name representation the tree is associated with, and we 
		 * specify the visit of the tree is not limited. If we had put 
		 * TreeNodePruner.UNLIMITED_RECURSION as in the previous pruning example
		 * we would have reduced the maximum number of children for each node. 
		 */
		TreeNodePruner maxChildrenPrunerObject = new TreeNodePruner(maxChildrenPruner, 
				representation, null, 1);
		return maxChildrenPrunerObject;
	}
	
	
}
