/*
 * Copyright 2015 Simone Filice and Giuseppe Castellucci and Danilo Croce and Roberto Basili
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

package it.uniroma2.sag.kelp.examples.demo.regression;

import java.util.Random;

import it.uniroma2.sag.kelp.data.dataset.SimpleDataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.label.Label;
import it.uniroma2.sag.kelp.data.label.StringLabel;
import it.uniroma2.sag.kelp.kernel.Kernel;
import it.uniroma2.sag.kelp.kernel.cache.FixSizeKernelCache;
import it.uniroma2.sag.kelp.kernel.standard.RbfKernel;
import it.uniroma2.sag.kelp.kernel.vector.LinearKernel;
import it.uniroma2.sag.kelp.learningalgorithm.regression.libsvm.EpsilonSvmRegression;
import it.uniroma2.sag.kelp.predictionfunction.Prediction;
import it.uniroma2.sag.kelp.predictionfunction.regressionfunction.RegressionFunction;
import it.uniroma2.sag.kelp.utils.evaluation.RegressorEvaluator;

/**
 * This class contains an example of the usage of the Regression Example. The
 * regressor implements the \(\epsilon\)-SVR learning algorithm discussed in [CC Chang, CJ Lin, 2011].
 * In this example a dataset is loaded from a file and then split in train and
 * test.
 * The dataset used in this example is the MG dataset. It can be downloaded
 * from:
 * http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/mg_scale
 * 
 * @author Danilo Croce
 */
public class EpsilonSVRegressionExample {

	public static void main(String[] args) throws Exception {
		// The epsilon in loss function of the regressor
		float pReg = 0.1f;
		// The regularization parameter of the regressor
		float c = 2f;
		// The gamma parameter in the RBF kernel
		float gamma = 1f;

		// The label indicating the value considered by the regressor
		Label label = new StringLabel("r");

		// Load the dataset
		SimpleDataset dataset = new SimpleDataset();
		dataset.populate("src/main/resources/sv_regression_test/mg_scale.klp");
		// Split the dataset in train and test datasets
		dataset.shuffleExamples(new Random(0));
		SimpleDataset[] split = dataset.split(0.7f);
		SimpleDataset trainDataset = split[0];
		SimpleDataset testDataset = split[1];

		// Kernel for the first representation (0-index)
		Kernel linear = new LinearKernel("0");
		// Applying the RBF kernel
		Kernel rbf = new RbfKernel(gamma, linear);
		// Applying a cache
		FixSizeKernelCache kernelCache = new FixSizeKernelCache(
				trainDataset.getNumberOfExamples());
		rbf.setKernelCache(kernelCache);

		// instantiate the regressor
		EpsilonSvmRegression regression = new EpsilonSvmRegression(rbf, label,
				c, pReg);

		// learn
		regression.learn(trainDataset);
		// get the prediction function
		RegressionFunction regressor = regression.getPredictionFunction();

		// initializing the performance evaluator
		RegressorEvaluator evaluator = new RegressorEvaluator(
				trainDataset.getRegressionProperties());

		// For each example from the test set
		for (Example e : testDataset.getExamples()) {
			// Predict the value
			Prediction prediction = regressor.predict(e);
			// Print the original and the predicted values
			System.out.println("real value: " + e.getRegressionValue(label)
					+ "\t-\tpredicted value: " + prediction.getScore(label));
			// Update the evaluator
			evaluator.addCount(e, prediction);
		}

		// Get the Mean Squared Error for the targeted label
		float measSquareError = evaluator.getMeanSquaredError(label);

		System.out.println("\nMean Squared Error:\t" + measSquareError);
	}

}
