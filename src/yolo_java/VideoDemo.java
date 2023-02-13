package yolo_java;


import org.opencv.core.Mat;
import org.opencv.highgui.HighGui;
import org.opencv.videoio.VideoCapture;


public class VideoDemo {
	
	public static void main(String[] args) throws InterruptedException {
	    String filePath = "test.mp4"; // video file to be analyzed
		float threshold = 0.6f;
		String configFile = "yolov3.cfg";
		String weightsFile = "yolov3.weights";
		String namesFile = "coco.names";

		Vision vision = new Vision(configFile, weightsFile, namesFile, threshold);
	    VideoCapture cap = new VideoCapture(filePath);// Load video using the videocapture method
	    Mat frame = new Mat(); // define a matrix to extract and store pixel info from video

	    while (cap.read(frame)) {
	    	Composition scene = vision.detectThings(frame);
	    	scene = scene.reduceOverlap(threshold);
	    	Mat outputFrame = scene.drawBoxes();
	    	HighGui.imshow("Video", outputFrame);
	    	HighGui.waitKey(1); // imshow requires this event handler to display window
	    }
	}     
		
}