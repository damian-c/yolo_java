package yolo_java;


import org.opencv.core.Mat;
import org.opencv.highgui.HighGui;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;


public class CameraDemo {
	private static Composition scene = new Composition(); // scene object shared between threads

	public static void main(String[] args) throws InterruptedException {
		int camera = 0; // which camera to use
		float threshold = 0.6f;
		String configFile = "yolov3.cfg";
		String weightsFile = "yolov3.weights";
		String namesFile = "coco.names";
		
		Vision vision = new Vision(configFile, weightsFile, namesFile, threshold);	
	    VideoCapture cap = new VideoCapture(camera);	
	    cap.set(Videoio.CAP_PROP_FRAME_WIDTH, 640);
	    cap.set(Videoio.CAP_PROP_FRAME_HEIGHT, 480);
		Mat frame = new Mat(); // define a matrix to extract and store pixel info from video
	    
		// do the processing in a separate thread
		class VisualProcessing implements Runnable {
			public void run() {
				while (true) {
					Composition newScene = vision.detectThings(frame);
					scene = newScene.reduceOverlap(threshold);
				}
			}
		}
	    Thread visionThread = new Thread(new VisualProcessing());
	    visionThread.start();
	    
	    while (cap.read(frame)) {
	    	scene.setImage(frame);
	    	Mat outputFrame = scene.drawBoxes();
	    	HighGui.imshow("Camera", outputFrame);
	    	HighGui.waitKey(1); // imshow requires this event handler to display window
	    }
	}     
}
