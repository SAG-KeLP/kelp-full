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

package it.uniroma2.sag.kelp.examples.demo.tweetsent2013;

import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;

import it.uniroma2.sag.kelp.data.dataset.Dataset;
import it.uniroma2.sag.kelp.data.dataset.SimpleDataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.label.Label;
import it.uniroma2.sag.kelp.kernel.Kernel;
import it.uniroma2.sag.kelp.kernel.cache.FixIndexSquaredNormCache;
import it.uniroma2.sag.kelp.kernel.cache.FixSizeKernelCache;
import it.uniroma2.sag.kelp.kernel.standard.LinearKernelCombination;
import it.uniroma2.sag.kelp.kernel.standard.NormalizationKernel;
import it.uniroma2.sag.kelp.kernel.standard.PolynomialKernel;
import it.uniroma2.sag.kelp.kernel.standard.RbfKernel;
import it.uniroma2.sag.kelp.kernel.vector.LinearKernel;
import it.uniroma2.sag.kelp.learningalgorithm.classification.libsvm.BinaryCSvmClassification;
import it.uniroma2.sag.kelp.learningalgorithm.classification.multiclassification.OneVsAllLearning;
import it.uniroma2.sag.kelp.predictionfunction.classifier.multiclass.OneVsAllClassificationOutput;
import it.uniroma2.sag.kelp.predictionfunction.classifier.multiclass.OneVsAllClassifier;
import it.uniroma2.sag.kelp.utils.JacksonSerializerWrapper;
import it.uniroma2.sag.kelp.utils.ObjectSerializer;
import it.uniroma2.sag.kelp.utils.evaluation.MulticlassClassificationEvaluator;
import it.uniroma2.sag.kelp.utils.exception.NoSuchPerformanceMeasureException;

public class TweetSentimentAnalysisSemeval2013 {
	private static String FIELD_SEP = "\t";
	private static String errors_file = "src/main/resources/tweetSentiment2013/errors.txt";

	public static void main(String[] args) throws Exception {
		float split = 0.8f;
		String train_file = "src/main/resources/tweetSentiment2013/train.klp.gz";
		String test_file = "src/main/resources/tweetSentiment2013/test.klp.gz";
		int kernelmode = 1;
		float polyD = 0;
		float gamma = 0;

		float[] Cs = new float[] { 0.1f, 0.5f, 1f };

		// Read a dataset into a test variable
		SimpleDataset trainingSet = new SimpleDataset();
		trainingSet.populate(train_file);
		// Read a dataset into a test variable
		SimpleDataset testSet = new SimpleDataset();
		testSet.populate(test_file);
		// set the cache size
		int cacheSize = trainingSet.getNumberOfExamples()
				+ testSet.getNumberOfExamples();
		// Initialize a kernel
		Kernel kernel = null;
		switch (kernelmode) {
		case 1:
			kernel = getBowKernel(cacheSize);
			break;
		case 2:
			kernel = getPolyBow(cacheSize, polyD);
			break;
		case 3:
			kernel = getWordspaceKernel(cacheSize);
			break;
		case 4:
			kernel = getRbfWordspaceKernel(cacheSize, gamma);
			break;
		case 5:
			kernel = getBowWordSpaceKernel(cacheSize);
			break;
		case 6:
			kernel = getPolyBowRbfWordspaceKernel(cacheSize, polyD, gamma);
			break;
		default:
			kernel = getBowKernel(cacheSize);
			break;
		}

		// Find optimal C
		float c = tune(trainingSet, kernel, split, Cs);
		System.out.println("start testing with C=" + c);
		// test
		float f1 = test(trainingSet, kernel, c, testSet, true);
		System.out.println("Mean F1 on test set=" + f1);
	}

	private static float test(SimpleDataset trainingSet, Kernel kernel,
			float c, SimpleDataset testSet, boolean printErrors)
			throws NoSuchPerformanceMeasureException, IOException {
		ArrayList<Label> classes = (ArrayList<Label>) trainingSet
				.getClassificationLabels();

		// Instantiate an svmSolver
		BinaryCSvmClassification svmSolver = new BinaryCSvmClassification();
		svmSolver.setKernel(kernel);
		svmSolver.setCp(c);
		svmSolver.setCn(c);
		svmSolver.setFairness(true);

		// Instantiate a OneVsAll multiclassification schema
		OneVsAllLearning ovaLearner = new OneVsAllLearning();
		ovaLearner.setBaseAlgorithm(svmSolver);
		ovaLearner.setLabels(classes);
		// Learn
		ovaLearner.learn(trainingSet);

		// Get and save on file the learning and prediction Function
		OneVsAllClassifier f = ovaLearner.getPredictionFunction();
		ObjectSerializer serializer = new JacksonSerializerWrapper();
		serializer
				.writeValueOnFile(ovaLearner,
						"src/main/resources/tweetSentiment2013/learningAlgorithmSpecification_multi.klp");
		serializer
				.writeValueOnFile(f,
						"src/main/resources/tweetSentiment2013/classificationAlgorithm_bow_ws.klp");

		// Adopt a built-in evaluator
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator(
				classes);
		PrintStream ps = null;
		if (printErrors)
			ps = new PrintStream(errors_file, "UTF-8");
		for (Example e : testSet.getExamples()) {
			OneVsAllClassificationOutput predict = f.predict(e);
			Label gold = e.getLabels()[0];
			Label pred = predict.getPredictedClasses().get(0);
			if (printErrors)
				ps.println(gold + "\t" + pred + "\t"
						+ (gold.equals(pred) ? "1" : "0"));

			evaluator.addCount(e, predict);
		}
		if (printErrors) {
			ps.flush();
			ps.close();
		}
		Label neu = findLabel("neutral", classes);
		Label pos = findLabel("positive", classes);
		Label neg = findLabel("negative", classes);
		ArrayList<Label> posNeg = new ArrayList<Label>();
		posNeg.add(pos);
		posNeg.add(neg);

		ArrayList<Label> posNegNeu = new ArrayList<Label>();
		posNegNeu.add(pos);
		posNegNeu.add(neg);
		posNegNeu.add(neu);

		StringBuilder b = new StringBuilder();
		for (Label l : posNegNeu) {
			b.append(FIELD_SEP + l + FIELD_SEP);
		}
		b.append("\n");
		b.append("Precision" + FIELD_SEP + "Recall" + FIELD_SEP + "F1"
				+ FIELD_SEP);
		b.append("Precision" + FIELD_SEP + "Recall" + FIELD_SEP + "F1"
				+ FIELD_SEP);
		b.append("Precision" + FIELD_SEP + "Recall" + FIELD_SEP + "F1"
				+ FIELD_SEP + "F1-Pn" + FIELD_SEP + "F1-Pnn" + "\n");
		for (Label l : posNegNeu) {
			b.append(evaluator.getPrecisionFor(l) + FIELD_SEP
					+ evaluator.getRecallFor(l) + FIELD_SEP
					+ evaluator.getF1For(l) + FIELD_SEP);
		}

		Object[] args = new Object[1];
		args[0] = posNeg;

		b.append(evaluator.getPerformanceMeasure("MeanF1For", args) + FIELD_SEP);
		b.append(evaluator.getPerformanceMeasure("MeanF1"));

		System.out.println(b.toString());
		return evaluator.getMeanF1();
	}

	private static Label findLabel(String string, List<Label> classes) {
		for (Label l : classes) {
			if (l.toString().equalsIgnoreCase(string))
				return l;
		}
		return null;
	}

	private static float tune(SimpleDataset allTrainingSet, Kernel kernel,
			float split, float[] cs) throws NoSuchPerformanceMeasureException,
			IOException {
		float bestC = 0.0f;
		float bestF1 = -Float.MAX_VALUE;

		// Split data according to a fix split
		Dataset[] split2 = allTrainingSet
				.splitClassDistributionInvariant(split);
		SimpleDataset trainingSet = (SimpleDataset) split2[0];
		SimpleDataset testSet = (SimpleDataset) split2[1];
		// tune parameter C
		for (float c : cs) {
			float f1 = test(trainingSet, kernel, c, testSet, false);
			System.out.println("C:" + c + "\t" + f1);
			if (f1 > bestF1) {
				bestF1 = f1;
				bestC = c;
			}
		}

		return bestC;
	}

	private static Kernel getBowKernel(int cacheSize) {
		Kernel kernel = new LinearKernel("BOW");
		FixIndexSquaredNormCache normcache = new FixIndexSquaredNormCache(
				cacheSize);
		kernel.setSquaredNormCache(normcache);

		kernel = new NormalizationKernel(kernel);
		FixSizeKernelCache cache = new FixSizeKernelCache(cacheSize);
		kernel.setKernelCache(cache);

		return kernel;
	}

	private static Kernel getPolyBow(int cacheSize, float polyD) {
		Kernel kernel1 = new LinearKernel("BOW");
		kernel1 = new PolynomialKernel(polyD, kernel1);
		FixIndexSquaredNormCache normcache1 = new FixIndexSquaredNormCache(
				cacheSize);
		kernel1.setSquaredNormCache(normcache1);
		kernel1 = new NormalizationKernel(kernel1);

		FixSizeKernelCache cache = new FixSizeKernelCache(cacheSize);
		kernel1.setKernelCache(cache);

		return kernel1;
	}

	private static Kernel getWordspaceKernel(int cacheSize) {
		Kernel kernel2 = new LinearKernel("WS");
		FixIndexSquaredNormCache normcache2 = new FixIndexSquaredNormCache(
				cacheSize);
		kernel2.setSquaredNormCache(normcache2);
		kernel2 = new NormalizationKernel(kernel2);

		FixSizeKernelCache cache = new FixSizeKernelCache(cacheSize);
		kernel2.setKernelCache(cache);

		return kernel2;
	}

	private static Kernel getRbfWordspaceKernel(int cacheSize, float gamma) {
		Kernel kernel2 = new LinearKernel("WS");
		kernel2 = new RbfKernel(gamma, kernel2);
		FixIndexSquaredNormCache normcache2 = new FixIndexSquaredNormCache(
				cacheSize);
		kernel2.setSquaredNormCache(normcache2);
		kernel2 = new NormalizationKernel(kernel2);

		FixSizeKernelCache cache = new FixSizeKernelCache(cacheSize);
		kernel2.setKernelCache(cache);

		return kernel2;
	}

	private static Kernel getBowWordSpaceKernel(int cacheSize) {
		Kernel kernel1 = new LinearKernel("BOW");
		FixIndexSquaredNormCache normcache1 = new FixIndexSquaredNormCache(
				cacheSize);
		kernel1.setSquaredNormCache(normcache1);
		kernel1 = new NormalizationKernel(kernel1);

		Kernel kernel2 = new LinearKernel("WS");
		FixIndexSquaredNormCache normcache2 = new FixIndexSquaredNormCache(
				cacheSize);
		kernel2.setSquaredNormCache(normcache2);
		kernel2 = new NormalizationKernel(kernel2);

		LinearKernelCombination combination = new LinearKernelCombination();
		combination.addKernel(1.0f, kernel1);
		combination.addKernel(1.0f, kernel2);

		FixSizeKernelCache cache = new FixSizeKernelCache(cacheSize);
		combination.setKernelCache(cache);

		return combination;
	}

	private static Kernel getPolyBowRbfWordspaceKernel(int cacheSize,
			float polyD, float gamma) {
		Kernel kernel1 = new LinearKernel("BOW");
		kernel1 = new PolynomialKernel(polyD, kernel1);
		FixIndexSquaredNormCache normcache1 = new FixIndexSquaredNormCache(
				cacheSize);
		kernel1.setSquaredNormCache(normcache1);
		kernel1 = new NormalizationKernel(kernel1);

		Kernel kernel2 = new LinearKernel("WS");
		kernel2 = new RbfKernel(gamma, kernel2);
		FixIndexSquaredNormCache normcache2 = new FixIndexSquaredNormCache(
				cacheSize);
		kernel2.setSquaredNormCache(normcache2);
		kernel2 = new NormalizationKernel(kernel2);

		LinearKernelCombination combination = new LinearKernelCombination();
		combination.addKernel(1.0f, kernel1);
		combination.addKernel(1.0f, kernel2);

		FixSizeKernelCache cache = new FixSizeKernelCache(cacheSize);
		combination.setKernelCache(cache);

		return combination;
	}
}
