using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System.Drawing;

namespace HandGestureConsoleApp
{
	class Program
	{
		// Drawing state
		private static bool _drawing = false;
		private static Bitmap _canvas;
		private static Graphics _graphics;
		private static Color _drawColor = Color.Black;
		private static Point _prevPoint = Point.Empty;

		static void Main(string[] args)
		{
			// Initialize webcam capture
			using (VideoCapture capture = new VideoCapture(0))
			{
				// Create a canvas for drawing
				_canvas = new Bitmap(640, 480);  // Match your webcam resolution
				_graphics = Graphics.FromImage(_canvas);

				while (true)
				{
					using (Mat frame = capture.QueryFrame())
					{
						if (frame == null) break;  // If no frame, exit

						// Convert the frame to grayscale for hand detection
						Image<Gray, byte> grayFrame = frame.ToImage<Gray, byte>();

						// Detect hands and gestures
						DetectHandsAndGestures(grayFrame, frame);

						CvInvoke.Imshow("Live Hand Gesture Detection", frame);

						// Press 'q' to exit the application
						if (CvInvoke.WaitKey(1) == 'q')
						{
							break;
						}
					}
				}
			}

			// Save the final canvas image when exiting
			SaveImage();
			Console.WriteLine("Drawing saved to disk. Exiting...");
		}

		private static void DetectHandsAndGestures(Image<Gray, byte> grayFrame, Mat frame)
		{
			// Threshold the grayscale frame to find hands
			Image<Gray, byte> thresholdFrame = grayFrame.ThresholdBinary(new Gray(100), new Gray(255));

			// Find contours of hands
			VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
			CvInvoke.FindContours(thresholdFrame, contours, null, RetrType.External, ChainApproxMethod.ChainApproxSimple);

			for (int i = 0; i < contours.Size; i++)
			{
				// Access the individual contour
				VectorOfPoint contour = contours[i];

				// Filter out small contours based on area (e.g., keep only large contours)
				double contourArea = CvInvoke.ContourArea(contour);
				if (contourArea < 1000)  // Adjust this threshold as per your needs
				{
					continue;  // Skip small contours
				}

				// Optionally, simplify the contour with ApproxPolyDP to avoid self-intersections
				VectorOfPoint approxContour = new VectorOfPoint();
				CvInvoke.ApproxPolyDP(contour, approxContour, 3, true);  // Simplify the contour

				// Calculate convex hull
				VectorOfInt hull = new VectorOfInt();
				CvInvoke.ConvexHull(approxContour, hull, false);

				// Calculate convexity defects
				Mat defects = new Mat();
				try
				{
					CvInvoke.ConvexityDefects(approxContour, hull, defects);

					// Count fingers using convexity defects
					int fingerCount = CountFingers(defects, approxContour);

					// Handle left hand gestures based on finger count
					HandleLeftHandGestures(fingerCount);

					// If drawing mode is enabled, draw on the canvas using the right hand's position
					if (_drawing)
					{
						Point rightHandPoint = GetRightHandPosition(approxContour);  // Get the right hand's centroid position
						DrawOnCanvas(rightHandPoint);
					}
				}
				catch (Emgu.CV.Util.CvException ex)
				{
					Console.WriteLine($"Convexity defect calculation failed: {ex.Message}");
				}
			}
		}


		private static int CountFingers(Mat defects, VectorOfPoint contour)
		{
			int fingerCount = 0;

			if (defects.Rows > 0)
			{
				for (int i = 0; i < defects.Rows; i++)
				{
					var defectData = defects.GetRawData(i);  // Get the convexity defect data
					if (defectData != null)
					{
						// Count fingers based on convexity defects (basic implementation)
						fingerCount++;
					}
				}
			}

			return fingerCount;
		}

		private static void HandleLeftHandGestures(int fingerCount)
		{
			if (fingerCount == 1)
			{
				_drawColor = Color.Black;  // Set color to black for drawing
			}
			else if (fingerCount == 0)
			{
				_drawing = false;  // Stop drawing
			}
			else if (fingerCount == 5)
			{
				_drawing = true;  // Start drawing
			}
			else if (fingerCount == 2)
			{
				SaveImage();  // Save the image if two fingers are shown
			}
		}

		private static Point GetRightHandPosition(VectorOfPoint contour)
		{
			// Calculate the centroid (center of mass) of the contour to get the hand's position
			var moments = CvInvoke.Moments(contour);

			int cX = (int)(moments.M10 / moments.M00);
			int cY = (int)(moments.M01 / moments.M00);

			return new Point(cX, cY);
		}

		private static void DrawOnCanvas(Point currentPoint)
		{
			if (_prevPoint.IsEmpty)
			{
				_prevPoint = currentPoint;
			}

			using (Pen pen = new Pen(_drawColor, 5))
			{
				_graphics.DrawLine(pen, _prevPoint, currentPoint);
			}

			_prevPoint = currentPoint;
		}

		private static void SaveImage()
		{
			string savePath = @"C:\Users\arka\Desktop\painting.png";
			_canvas.Save(savePath);
			Console.WriteLine($"Image saved to {savePath}");
		}
	}
}
