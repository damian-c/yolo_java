package yolo_java;


import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;


public class ImageDemo {

	public static void main(String[] args) {
		String filePath = "test.jpg"; // image file to be analyzed
		float threshold = 0.6f;
		String configFile = "yolov3.cfg";
		String weightsFile = "yolov3.weights";
		String namesFile = "coco.names";

		Vision vision = new Vision(configFile, weightsFile, namesFile, threshold);
		Mat image = Imgcodecs.imread(filePath);
		Composition scene = vision.detectThings(image);
		Mat outputImage = scene.reduceOverlap(threshold).drawBoxes();
		Imgcodecs.imwrite("out.jpg", outputImage);
	}

}
