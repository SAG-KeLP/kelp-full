package it.uniroma2.sag.kelp.data.dataset;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.StringWriter;

import org.junit.Assert;
import org.junit.Test;

import com.fasterxml.jackson.databind.ObjectMapper;

import it.uniroma2.sag.kelp.kernel.Kernel;
import it.uniroma2.sag.kelp.kernel.cache.FixIndexKernelCache;
import it.uniroma2.sag.kelp.kernel.cache.FixIndexSquaredNormCache;
import it.uniroma2.sag.kelp.kernel.standard.PolynomialKernel;
import it.uniroma2.sag.kelp.kernel.vector.LinearKernel;
import it.uniroma2.sag.kelp.wordspace.Wordspace;

public class FileUtilsTest {

	@Test
	public void readGzippedDataset() throws Exception {
		SimpleDataset d = new SimpleDataset();
		d.populate("src/test/resources/dataset_reader_test.klp.gz");
		Assert.assertTrue(true);
	}

	@Test
	public void checkKernelGzipped() throws FileNotFoundException, IOException {
		Kernel a = new LinearKernel("REPR1");
		FixIndexSquaredNormCache cache = new FixIndexSquaredNormCache(10);
		a.setSquaredNormCache(cache);
		a = new PolynomialKernel(2f, a);
		a.setKernelCache(new FixIndexKernelCache(10));

		Kernel.save(a, "src/test/resources/kernel.gz");
		Assert.assertTrue(true);

		Kernel b = Kernel.load("src/test/resources/kernel.gz");
		ObjectMapper mapper = new ObjectMapper();

		StringWriter w1 = new StringWriter();
		mapper.writeValue(w1, a);

		StringWriter w2 = new StringWriter();
		mapper.writeValue(w2, b);

		System.out.println(w1.toString());
		System.out.println(w2.toString());

		Assert.assertTrue(w1.toString().equals(w2.toString()));

		File f = new File("src/test/resources/kernel.gz");
		f.delete();
	}

	@Test
	public void checkLoadWordspace() throws IOException {
		Wordspace ws = new Wordspace(
				"src/main/resources/wordspace/wordspace_qc.txt.gz");
		Assert.assertTrue(true);
	}
}
