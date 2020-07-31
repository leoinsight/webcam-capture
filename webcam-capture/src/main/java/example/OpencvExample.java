package example;

import nu.pattern.OpenCV;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

public class OpencvExample {

    public static void main(String[] args) {
        // load opencv library
        OpenCV.loadShared();
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

//        Mat mat = Mat.eye(3, 3, CvType.CV_8UC1);
//        System.out.println("mat = " + mat.dump());

        // finger detection with hsv
        String savePath = "src/main/resources";
        String imagePath = new File(savePath + "/gongcha_menu.png").getAbsolutePath();
        // convert source image to HSV
        Mat srcImage = loadImage(imagePath);
        Mat blurImage = new Mat();
        Mat hsvImage = new Mat();
        Imgproc.blur(srcImage, blurImage, new Size(7, 7));  // remove some noise
        Imgproc.cvtColor(blurImage, hsvImage, Imgproc.COLOR_BGR2HSV);
        // mask hsv image
        Scalar scalarLower = new Scalar(0, 0.28*255, 0);
        Scalar scalarUpper = new Scalar(25, 0.68*255, 255);
        Mat maskedImage = new Mat();
        Core.inRange(hsvImage, scalarLower, scalarUpper, maskedImage);
        // save masked image
        saveImage(maskedImage, savePath + "/gongcha_menu_detected.png");

        // morphological operators
        // dilate with large element, erode with small ones
        Mat morphImage = new Mat();
        Mat dilateElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(12, 12));
        Mat erodeElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(12, 12));
        Imgproc.erode(maskedImage, morphImage, erodeElement);
        Imgproc.erode(maskedImage, morphImage, erodeElement);
        Imgproc.dilate(maskedImage, morphImage, dilateElement);
        Imgproc.dilate(maskedImage, morphImage, dilateElement);

        // find contours
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(morphImage, contours, hierarchy, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE);
        // draw contours
        Mat contImage = srcImage.clone();
        Imgproc.drawContours(contImage, contours,-1, new Scalar(0, 255, 0), 2);
        saveImage(contImage, savePath + "/gongcha_menu_contour.png");

        // draw convex hull
        MatOfPoint points = contours.get(0);
        List<MatOfPoint> hullList = new ArrayList<>();
        for (MatOfPoint contour : contours) {
            MatOfInt hull = new MatOfInt();
            Imgproc.convexHull(points, hull);
            Point[] contourArray = points.toArray();
            Point[] hullPoints = new Point[hull.rows()];
            List<Integer> hullContourIdxList = hull.toList();
            for (int i = 0; i < hullContourIdxList.size(); i++) {
                hullPoints[i] = contourArray[hullContourIdxList.get(i)];
            }
            hullList.add(new MatOfPoint(hullPoints));
        }
        Mat convImage = contImage.clone();
        Imgproc.drawContours(convImage, hullList, -1, new Scalar(0, 0, 255), 2);
        saveImage(convImage, savePath + "/gongcha_menu_convex_hull.png");
    }

    // load image
    public static Mat loadImage(String imagePath) {
        Imgcodecs imageCodecs = new Imgcodecs();
        return imageCodecs.imread(imagePath);
    }
    // save image
    public static void saveImage(Mat imageMatrix, String targetPath) {
        Imgcodecs imgcodecs = new Imgcodecs();
        imgcodecs.imwrite(targetPath, imageMatrix);
    }

}
