import koboldNet.KoboldNet;

/**
 * Mr.Kobold
 * Main Class, entry point of the program
 */
public class DerMain {

	public static void main(String[] args) {

		KoboldNet koboldNet = new KoboldNet();
		koboldNet.exportToFile("test.txt");

		KoboldNet koboldNet1 = KoboldNet.importFromFile("test.txt");
	}

}
