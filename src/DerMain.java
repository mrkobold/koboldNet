import koboldNet.IOUtil;
import koboldNet.KoboldNet;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;

/**
 * Mr.Kobold
 * Main Class, entry point of the program
 */

/**
 * TODO
 * forward propagate a batch & sum costs
 * bp based on this unified shite
 */
public class DerMain {

	private static final int NUM_TRAINING_BATCHES = 600;
	private static final int NUM_TESTING_BATCHES = 100;

	public static void main(String[] args) throws FileNotFoundException {

		KoboldNet koboldNet = IOUtil.importNetFromFile("testTraining.txt");

//		for (int i = 0; i < 3; i++) {
//			System.out.println("abcd" + i);
//			train(koboldNet);
//		}
		test(koboldNet);
	}

	private static void train(KoboldNet koboldNet) throws FileNotFoundException {
		DataInputStream imagesStream = new DataInputStream(new FileInputStream(new File("/home/boldizsarkovacs/koboldNetInputs/train-images.idx3-ubyte")));
		DataInputStream labelsStream = new DataInputStream(new FileInputStream(new File("/home/boldizsarkovacs/koboldNetInputs/train-labels.idx1-ubyte")));
		List<Double[]> inputs;
		List<Double[]> labels;
		for (int i = 0; i < NUM_TRAINING_BATCHES; i++) {
			inputs = new ArrayList<>();
			labels = new ArrayList<>();
			IOUtil.readBatch(inputs, labels,
							 imagesStream,
							 labelsStream);

			// train
			koboldNet.trainNetwork(inputs, labels);

			// export
			IOUtil.exportNetToFile("testTraining.txt", koboldNet);
		}
	}

	private static void test(KoboldNet koboldNet) throws FileNotFoundException {
		DataInputStream imagesStream = new DataInputStream(new FileInputStream(new File("/home/boldizsarkovacs/koboldNetInputs/t10k-images.idx3-ubyte")));
		DataInputStream labelsStream = new DataInputStream(new FileInputStream(new File("/home/boldizsarkovacs/koboldNetInputs/t10k-labels.idx1-ubyte")));
		List<Double[]> inputs;
		List<Double[]> labels;
		int correct;
		for (int i = 0; i < NUM_TESTING_BATCHES; i++) {
			inputs = new ArrayList<>();
			labels = new ArrayList<>();
			IOUtil.readBatch(inputs, labels,
							 imagesStream,
							 labelsStream);

			// train
			correct = koboldNet.testNetwork(inputs, labels);
			System.out.println("precision: " + ((double)correct / 100) + "% (" + correct + ")");
		}
	}
}
