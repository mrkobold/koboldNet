import koboldNet.IOUtil;
import koboldNet.KoboldNet;

import java.util.ArrayList;
import java.util.List;

/**
 * Mr.Kobold
 * Main Class, entry point of the program
 */
public class DerMain {

	private static final int NUM_BATCHES = 100;

	public static void main(String[] args) {


		KoboldNet koboldNet = new KoboldNet();
		koboldNet.initialize();
		IOUtil.exportNetToFile("testTraining.txt", koboldNet);

		List<Double[]> inputs;
		List<Double[]> labels;
		for (int i = 0; i < NUM_BATCHES; i++) {
			inputs = new ArrayList<>();
			labels = new ArrayList<>();
			IOUtil.readBatch(inputs, labels,
							 "/home/boldizsarkovacs/koboldNetInputs/train-images.idx3-ubyte",
							 "/home/boldizsarkovacs/koboldNetInputs/train-labels.idx1-ubyte");

			// train
			koboldNet.trainNetwork(inputs, labels);

			// export
			IOUtil.exportNetToFile("testTraining.txt", koboldNet);
		}
	}
}
