/*
 * Copyright 2014 Simone Filice and Giuseppe Castellucci and Danilo Croce and Roberto Basili
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

package it.uniroma2.sag.kelp.examples.demo.rcv1;

import it.uniroma2.sag.kelp.data.dataset.SimpleDataset;
import it.uniroma2.sag.kelp.data.label.StringLabel;
import it.uniroma2.sag.kelp.learningalgorithm.LearningAlgorithm;
import it.uniroma2.sag.kelp.learningalgorithm.classification.dcd.DCDLearningAlgorithm;
import it.uniroma2.sag.kelp.learningalgorithm.classification.dcd.DCDLoss;

public class RCV1BinaryTextCategorizationDCD extends RCV1BinaryTextCategorization {
	protected String algoSuffix = "LibLinear";
	
	public static void main(String[] args) {
		System.setProperty("org.slf4j.simpleLogger.defaultLogLevel", "WARN");

		String train_file = "src/main/resources/rcv1/rcv1_train_liblsite.klp.gz";
		float c = 1f;
		int nfold = 5;

		SimpleDataset allData = new SimpleDataset();
		try {
			allData.populate(train_file);
		} catch (Exception e) {
			e.printStackTrace();
		}

		RCV1BinaryTextCategorization foldLearning = new RCV1BinaryTextCategorizationDCD();
		foldLearning.foldLearn(c, nfold, allData);
	}
	
	@Override
	protected LearningAlgorithm getLearningAlgorithm(float param, String representation, StringLabel positiveLabel) {		
		/**
		 * The considered Loss function (L1 or L2)
		 */
		DCDLoss dcdLoss = DCDLoss.L2;
		/**
		 * This boolean parameter determines the use of bias <code>b</code> in the
		 * classification function <cod>f(x)=wx+b</code>. If usebias is set to
		 * <code>false</code> the bias is set to 0.
		 */
		boolean usebias = true;
		/**
		 * The number of iteration of the main algorithm
		 */
		int iterations = 20;

		DCDLearningAlgorithm algo = new DCDLearningAlgorithm(param, param,
				dcdLoss, usebias, iterations, representation);
		algo.setLabel(positiveLabel);
		
		return algo;
	}
}
