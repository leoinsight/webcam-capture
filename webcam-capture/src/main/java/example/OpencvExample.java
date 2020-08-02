package example;

import ch.qos.logback.core.net.SyslogOutputStream;
import nu.pattern.OpenCV;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

import static org.opencv.core.CvType.CV_32FC2;

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
        Scalar scalarLower = new Scalar(0, 30, 0);
        Scalar scalarUpper = new Scalar(15, 255, 255);
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

        // get finger contour
        int centerX = srcImage.width() / 2;
        int bottomY = srcImage.height();
        Point pt = new Point(centerX, bottomY - 1);
        MatOfPoint fingerCont = getFingerContour(contours, pt);
        // draw contours
        Mat contourImage = srcImage.clone();
        List<MatOfPoint> contList = new ArrayList<>();
        contList.add(fingerCont);
        Imgproc.drawContours(contourImage, contList, -1, new Scalar(0, 255, 0), 2);
        saveImage(contourImage, savePath + "/gongcha_menu_contour.png");

//        // get convex hull
//        List<MatOfPoint> convexHull = getConvexHull(contours);
//        // draw convex hull
//        Mat convexImage = contourImage.clone();
//        Imgproc.drawContours(convexImage, convexHull, -1, new Scalar(0, 0, 255), 2);
//        saveImage(convexImage, savePath + "/gongcha_menu_convex_hull.png");
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

    /**
     * get a finger contour <br>
     *  - determines whether the point is inside a contour
     * @param contours
     * @return
     */
    public static MatOfPoint getFingerContour(List<MatOfPoint> contours, Point pt) {
        MatOfPoint fingerCont = new MatOfPoint();
        for (int i = 0; i < contours.size(); i++) {
            MatOfPoint cont = contours.get(i);
            MatOfPoint2f con2 = new MatOfPoint2f();
            cont.convertTo(con2, CV_32FC2);
            double pointInContour = Imgproc.pointPolygonTest(con2, pt, false);
            System.out.println("src image point(center,bottom-1): (" + pt.x + "," + pt.y + ")");
            System.out.println("point is inside or on edge?:" + (pointInContour > -1.0 ? "Yes" : "No"));
            if (pointInContour > -1) {
                System.out.println("pointInCountour index:" + i);
                fingerCont = cont;
                break;
            }
        }
        return fingerCont;
    }

    // find max contour
    public static MatOfPoint findMaxContour(List<MatOfPoint> contours) {
        MatOfPoint maxContour = new MatOfPoint();
        for (int i = 0; i < contours.size(); i++) {
            MatOfPoint contour = contours.get(i);

            Rect rect = Imgproc.boundingRect(contour);
            if(rect.width > rect.height) continue;

            if(i == 0) maxContour = contour;
            double contourArea = Imgproc.contourArea(contour);
            double maxContourArea = Imgproc.contourArea(maxContour);
            if(contourArea > maxContourArea) maxContour = contour;
        }
        return maxContour;
    }

    // get Convex hull
    public static List<MatOfPoint> getConvexHull(List<MatOfPoint> contours) {
        // draw convex hull
        List<MatOfPoint> hullList = new ArrayList<>();
        for (int i=0; i<contours.size(); i++) {
            MatOfPoint points = contours.get(i);
            MatOfInt hull = new MatOfInt();
            Imgproc.convexHull(points, hull);
            Point[] contourArray = points.toArray();
            Point[] hullPoints = new Point[hull.rows()];
            List<Integer> hullContourIdxList = hull.toList();
            for (int j = 0; j < hullContourIdxList.size(); j++) {
                hullPoints[j] = contourArray[hullContourIdxList.get(j)];
            }
            hullList.add(new MatOfPoint(hullPoints));
        }
        return hullList;
    }
}
