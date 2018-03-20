import koboldNet.KoboldNet;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Mr.Kobold
 * Main Class, entry point of the program
 */
public class DerMain {

	private static final int NUM_BATCHES = 100;

	public static void main(String[] args) {

		List<Double[]> inputs;
		List<Double[]> labels;

		KoboldNet koboldNet = new KoboldNet();
		koboldNet.initialize();
		koboldNet.exportToFile("testTraining.txt");

		for (int i = 0; i < NUM_BATCHES; i++) {
			inputs = new ArrayList<>();
			labels = new ArrayList<>();
			readBatch(inputs, labels,
					  "/home/boldizsarkovacs/koboldNetInputs/train-images.idx3-ubyte",
					  "/home/boldizsarkovacs/koboldNetInputs/train-labels.idx1-ubyte");

			// train
			koboldNet.trainNetwork(inputs, labels);

			// export
			koboldNet.exportToFile("testTraining.txt");
		}
	}

	private static void readBatch(List<Double[]> inputs, List<Double[]> labels, String imagesFile, String labelsFile) {
		try {
			DataInputStream imagesStream = new DataInputStream(new FileInputStream(new File(imagesFile)));
			DataInputStream labelsStream = new DataInputStream(new FileInputStream(new File(labelsFile)));
			for (int i = 0; i < 100; i++) {
				Double image[] = new Double[28 * 28];
				Double label[] = new Double[10];

				for (int j = 0; j < 28; j++) {
					for (int k = 0; k < 28; k++) {
						image[28 * j + k] = (double) imagesStream.readUnsignedByte();
					}
				}

				Arrays.fill(label, 0.0);
				label[labelsStream.readUnsignedByte()] = 1.0;

				inputs.add(image);
				labels.add(label);
			}
		} catch (Exception ex) {
			ex.printStackTrace();
		}
	}

}
