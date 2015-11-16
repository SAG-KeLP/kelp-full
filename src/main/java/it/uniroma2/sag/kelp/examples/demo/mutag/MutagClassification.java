/*
 * Copyright 2014-2015 Simone Filice and Giuseppe Castellucci and Danilo Croce and Roberto Basili
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

package it.uniroma2.sag.kelp.examples.demo.mutag;

import it.uniroma2.sag.kelp.data.dataset.SimpleDataset;
import it.uniroma2.sag.kelp.data.label.StringLabel;
import it.uniroma2.sag.kelp.data.manipulator.WLSubtreeMapper;
import it.uniroma2.sag.kelp.kernel.cache.FixIndexKernelCache;
import it.uniroma2.sag.kelp.kernel.graph.ShortestPathKernel;
import it.uniroma2.sag.kelp.kernel.standard.LinearKernelCombination;
import it.uniroma2.sag.kelp.kernel.vector.LinearKernel;
import it.uniroma2.sag.kelp.learningalgorithm.classification.libsvm.BinaryCSvmClassification;
import it.uniroma2.sag.kelp.utils.ExperimentUtils;
import it.uniroma2.sag.kelp.utils.evaluation.BinaryClassificationEvaluator;

import java.util.List;

/**
 * This code performs a 10-fold cross validation on the MUTAG dataset [1] 
 * training a C-SVM using a linear kernel combination the <code>ShortestPathKernel</code> (operating on DirectedGraphRepresentations)
 * and the Weisfeiler-Lehman Subtree Kernel for Graphs. This last is performed applying an explicit mapping of the
 * graphs into the RKHS of the kernel. This mapping is operated by the <code>WLSubtreeMapper</code>
 * 
 * [1] Debnath, A.K. Lopez de Compadre, R.L., Debnath, G., Shusterman, A.J., and Hansch, C. (1991). 
 * Structure-activity relationship of mutagenic aromatic and heteroaromatic nitro compounds. 
 * Correlation with molecular orbital energies and hydrophobicity. J. Med. Chem. 34:786-797.
 * 
 * @author Giovanni Da San martino, Simone Filice
 *
 */
public class MutagClassification {
	
	private static final String GRAPH_REPRESENTATION_NAME = "inline";
	private static final String VECTORIAL_LINEARIZATION_NAME = "wl";

	public static void main(String[] args) throws Exception {
		SimpleDataset trainingSet = new SimpleDataset();
		trainingSet.populate("src/main/resources/mutag/mutag.txt");
		StringLabel positiveClass = new StringLabel("1");

		System.out.println("Training set statistics");
		System.out.print("Examples number ");
		System.out.println(trainingSet.getNumberOfExamples());
		System.out.print("Positive examples ");
		System.out.println(trainingSet
				.getNumberOfPositiveExamples(positiveClass));
		System.out.print("Negative examples ");
		System.out.println(trainingSet
				.getNumberOfNegativeExamples(positiveClass));

		WLSubtreeMapper m = new WLSubtreeMapper(GRAPH_REPRESENTATION_NAME, VECTORIAL_LINEARIZATION_NAME, 4);
		trainingSet.manipulate(m);

		StringLabel targetLabel = new StringLabel("1");

		BinaryClassificationEvaluator evaluator = new BinaryClassificationEvaluator(targetLabel);
	    
	    LinearKernelCombination comb = new LinearKernelCombination();
	    LinearKernel linear = new LinearKernel(VECTORIAL_LINEARIZATION_NAME);
	    comb.addKernel(1, linear);
	    ShortestPathKernel spk = new ShortestPathKernel(GRAPH_REPRESENTATION_NAME);
	    comb.addKernel(1, spk);
	    comb.setKernelCache(new FixIndexKernelCache(trainingSet.getNumberOfExamples()));
	    BinaryCSvmClassification svmSolver = new BinaryCSvmClassification(comb, targetLabel, 1, 1);
		
	    float meanAcc = 0;
	    int nFold = 10;
	    List<BinaryClassificationEvaluator> evalutators = ExperimentUtils.nFoldCrossValidation(nFold, svmSolver, trainingSet, evaluator);
	    
		for(int i=0;i<nFold;i++){
			float accuracy = evalutators.get(i).getPerformanceMeasure("accuracy");
			System.out.println("fold " + (i+1) + " accuracy: " + accuracy);
			meanAcc+=accuracy;
		}
		
		meanAcc/=(float)10;
		System.out.println("MEAN ACC: " + meanAcc);
		
	}
	

}
