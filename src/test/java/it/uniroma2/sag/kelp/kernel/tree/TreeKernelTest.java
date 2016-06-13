/*
 * Copyright 2016 Simone Filice and Giuseppe Castellucci and Danilo Croce and Roberto Basili
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

package it.uniroma2.sag.kelp.kernel.tree;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.zip.GZIPInputStream;

import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;

import it.uniroma2.sag.kelp.data.dataset.SimpleDataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.manipulator.LexicalStructureElementManipulator;
import it.uniroma2.sag.kelp.data.representation.structure.similarity.LexicalStructureElementSimilarity;
import it.uniroma2.sag.kelp.data.representation.structure.similarity.compositional.sum.CompositionalNodeSimilaritySum;
import it.uniroma2.sag.kelp.kernel.Kernel;
import it.uniroma2.sag.kelp.kernel.standard.NormalizationKernel;
import it.uniroma2.sag.kelp.wordspace.Wordspace;

public class TreeKernelTest {

	private static final double TOLERANCE = 0;
	private static SimpleDataset testSet;
	private static Wordspace wordspace;

	public static void main(String[] args) throws Exception {

		SimpleDataset testSet = new SimpleDataset();
		testSet.populate("src/main/resources/qc/TREC_10.coarse.klp.gz");

		Kernel kernel = getQCKernelFunction(testSet, "csptk");
		ArrayList<Float> kernelScores = getKernelScores(testSet, kernel);
		PrintStream ps = new PrintStream("src/test/resources/kernels/tree/csptk_scores.txt");
		for (Float kernelScore : kernelScores) {
			ps.println(kernelScore);
		}
		ps.close();
	}

	@Test
	public void testPtk() {
		try {
			String filepath = "src/test/resources/kernels/tree/ptk_scores.txt";
			Kernel kernel = getQCKernelFunction(testSet, "ptk");

			ArrayList<Float> newKernelScores = getKernelScores(testSet, kernel);
			ArrayList<Float> oldKernelScores = loadKernelScores(filepath);

			double mse = 0f;
			for (int i = 0; i < newKernelScores.size(); ++i) {
				mse += Math.pow((double) (newKernelScores.get(i) - oldKernelScores.get(i)), 2.0);
			}
			mse /= (float) testSet.getExamples().size();
			Assert.assertEquals(0, mse, TOLERANCE);
		} catch (IOException e) {
			Assert.assertTrue(false);
		}
	}

	@Test
	public void testSptk() {
		try {
			String filepath = "src/test/resources/kernels/tree/sptk_scores.txt";
			Kernel kernel = getQCKernelFunction(testSet, "sptk");

			ArrayList<Float> newKernelScores = getKernelScores(testSet, kernel);
			ArrayList<Float> oldKernelScores = loadKernelScores(filepath);

			double mse = 0f;
			for (int i = 0; i < newKernelScores.size(); ++i) {
				mse += Math.pow((double) (newKernelScores.get(i) - oldKernelScores.get(i)), 2.0);
			}
			mse /= (float) testSet.getExamples().size();
			Assert.assertEquals(0, mse, TOLERANCE);
		} catch (IOException e) {
			Assert.assertTrue(false);
		}
	}

	@Test
	public void testCsptk() {
		try {
			String filepath = "src/test/resources/kernels/tree/csptk_scores.txt";
			Kernel kernel = getQCKernelFunction(testSet, "csptk");

			ArrayList<Float> newKernelScores = getKernelScores(testSet, kernel);
			ArrayList<Float> oldKernelScores = loadKernelScores(filepath);

			double mse = 0f;
			for (int i = 0; i < newKernelScores.size(); ++i) {
				mse += Math.pow((double) (newKernelScores.get(i) - oldKernelScores.get(i)), 2.0);
			}
			mse /= (float) testSet.getExamples().size();
			Assert.assertEquals(0, mse, TOLERANCE);
		} catch (IOException e) {
			Assert.assertTrue(false);
		}
	}

	@BeforeClass
	public static void loadDataset() throws Exception {
		testSet = new SimpleDataset();
		testSet.populate("src/main/resources/qc/TREC_10.coarse.klp.gz");
	}

	public static ArrayList<Float> loadKernelScores(String filepath) {
		try {
			ArrayList<Float> scores = new ArrayList<Float>();
			BufferedReader in = null;
			String encoding = "UTF-8";
			if (filepath.endsWith(".gz")) {
				in = new BufferedReader(
						new InputStreamReader(new GZIPInputStream(new FileInputStream(filepath)), encoding));
			} else {
				in = new BufferedReader(new InputStreamReader(new FileInputStream(filepath), encoding));
			}

			String str = "";
			while ((str = in.readLine()) != null) {
				scores.add(Float.parseFloat(str));
			}

			in.close();

			return scores;

		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
			Assert.assertTrue(false);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			Assert.assertTrue(false);
		} catch (IOException e) {
			e.printStackTrace();
			Assert.assertTrue(false);
		}

		return null;
	}

	private static ArrayList<Float> getKernelScores(SimpleDataset testSet, Kernel kernel) {
		ArrayList<Float> scores = new ArrayList<Float>();
		for (int i = 1; i < testSet.getNumberOfExamples(); i++) {
			Example exA = testSet.getExample(i - 1);
			Example exB = testSet.getExample(i);
			scores.add(kernel.innerProduct(exA, exB));
		}
		return scores;
	}

	public static Kernel getQCKernelFunction(SimpleDataset dataSet, String kernelId) throws IOException {

		Kernel usedKernel = null;
		/*
		 * Set the cache size
		 */

		if (kernelId.equalsIgnoreCase("ptk")) {
			// The representation on which the kernel operates
			String treeRepresentationName = "grct";
			// Kernel for the grct representation
			Kernel ptkgrct = new PartialTreeKernel(0.4f, 0.4f, 5f, treeRepresentationName);
			// The kernel is normalized.
			Kernel normPtkGrct = new NormalizationKernel(ptkgrct);
			usedKernel = normPtkGrct;
		} else if (kernelId.equalsIgnoreCase("sptk")) {
			// The representation on which the kernel operates
			String treeRepresentationName = "lct";
			// The node similarity function between lexical nodes is based
			// on a Distributional Model, as discussed in
			// [Croce et al.(2011)]
			String matrixPath = "src/main/resources/wordspace/wordspace_qc.txt.gz";
			// The word space containing the vector representation of words
			// represented in lexical nodes is loaded
			wordspace = new Wordspace(matrixPath);
			// This manipulator assigns vectors to lexical nodes. It allows
			// to speed-up computations: otherwise each time the similarity
			// between two nodes is evaluated, the corresponding vectors are
			// retrieved in the word space, with additional operational
			// costs.
			LexicalStructureElementManipulator lexManipulator = new LexicalStructureElementManipulator(wordspace,
					treeRepresentationName);
			dataSet.manipulate(lexManipulator);
			// This class implements a similarity function between lexical
			// nodes based on the Word space
			LexicalStructureElementSimilarity similarityWordspace = new LexicalStructureElementSimilarity(wordspace);
			// The kernel operating over the lct representation
			Kernel sptklct = new SmoothedPartialTreeKernel(0.4f, 0.4f, 0.2f, 0.01f, similarityWordspace,
					treeRepresentationName);
			// The kernel is normalized.
			NormalizationKernel normalizedSptkLct = new NormalizationKernel(sptklct);
			usedKernel = normalizedSptkLct;
		} else if (kernelId.equalsIgnoreCase("csptk")) {
			// The representation on which the kernel operates
			String treeRepresentationName = "clct";
			// The node similarity function between lexical nodes is based
			// on a Distributional Model, as in [Annesi et al.(2014)]
			String matrixPath = "src/main/resources/wordspace/wordspace_qc.txt.gz";
			wordspace = new Wordspace(matrixPath);
			// This manipulator assigns vectors to lexical nodes. It allows
			// to speed-up computations: otherwise each time the similarity
			// between two nodes is evaluated, the corresponding vectors are
			// retrieved in the word space, with additional operational
			// costs.
			LexicalStructureElementManipulator lexManipulator = new LexicalStructureElementManipulator(wordspace,
					treeRepresentationName);
			dataSet.manipulate(lexManipulator);
			// Compositional nodes syntactic nodes are represented as vector
			// that is the sum of the vector representing the syntactic head
			// and modifier, as discussed in [Annesi et al(2014)]
			CompositionalNodeSimilaritySum compSS = new CompositionalNodeSimilaritySum();
			compSS.setWordspace(wordspace);
			compSS.setRepresentationToBeEnriched(treeRepresentationName);
			// This manipulator assigns vectors to "compositional syntactic"
			// nodes. It allows to speed-up computations: otherwise each
			// time the similarity between two nodes is evaluated, the
			// corresponding vectors are retrieved in the word space, with
			// additional operational costs.
			dataSet.manipulate(compSS);
			// The kernel operating over the clct representation
			Kernel sptkcgrct = new SmoothedPartialTreeKernel(0.4f, 0.4f, 1f, 0.01f, compSS, treeRepresentationName);
			// The kernel is normalized.
			Kernel normSptklct = new NormalizationKernel(sptkcgrct);
			usedKernel = normSptklct;
		} else {
			System.err.println("The kernel " + kernelId + " has not been defined.");
		}
		return usedKernel;
	}

}
