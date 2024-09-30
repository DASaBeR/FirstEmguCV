using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System.Drawing;

namespace FirstEmguCV.ImageServices
{
	public class HandGestureDrawingApp
	{
		private VideoCapture _capture;
		private bool _drawing = false;
		private Point _prevPoint = new Point();
		private Bitmap _canvas;
		private Graphics _graphics;
		private Color _drawColor = Color.Black;

		public HandGestureDrawingApp()
		{
			// Initialize webcam capture
			_capture = new VideoCapture(0); // 0 is default camera
			_capture.ImageGrabbed += ProcessFrame;
			_capture.Start();

			// Create canvas
			_canvas = new Bitmap(640, 480);
			_graphics = Graphics.FromImage(_canvas);
			_graphics.Clear(Color.White);
		}

		private void ProcessFrame(object sender, EventArgs e)
		{
			Mat frame = new Mat();
			_capture.Retrieve(frame);

			// Convert frame to grayscale
			Image<Gray, byte> grayFrame = frame.ToImage<Gray, byte>();

			// Call hand detection and gesture recognition logic
			DetectHandsAndGestures(grayFrame, frame);

			// Show the frame with the drawn painting
			CvInvoke.Imshow("Hand Gesture Drawing", frame);
		}


		private void DetectHandsAndGestures(Image<Gray, byte> grayFrame, Mat frame)
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

				// Calculate convex hull
				VectorOfInt hull = new VectorOfInt();
				CvInvoke.ConvexHull(contour, hull, false);

				// Calculate convexity defects
				Mat defects = new Mat();
				CvInvoke.ConvexityDefects(contour, hull, defects);

				// Count fingers using convexity defects
				int fingerCount = CountFingers(defects, contour);

				// Check if left hand is showing specific gestures
				HandleLeftHandGestures(fingerCount);

				// If drawing is active, draw on canvas using the right hand's position
				if (_drawing)
				{
					Point rightHandPoint = GetRightHandPosition(contour);  // Calculate right hand's position
					DrawOnCanvas(rightHandPoint);
				}
			}
		}


		private int CountFingers(Mat defects, VectorOfPoint contour)
		{
			int fingerCount = 0;

			if (defects.Rows > 0)
			{
				for (int i = 0; i < defects.Rows; i++)
				{
					var defectData = defects.GetRawData(i);  // Get the convexity defect
					if (defectData != null)
					{
						int startIdx = Convert.ToInt32(defectData.GetValue(0)); // Starting point of defect
						int endIdx = Convert.ToInt32(defectData.GetValue(1));   // End point of defect
						int farIdx = Convert.ToInt32(defectData.GetValue(2));   // The farthest point

						// You can use these points to count fingers
						Point startPoint = contour[startIdx];
						Point endPoint = contour[endIdx];
						Point farthestPoint = contour[farIdx];

						// For each valid defect, we increase the finger count
						// You can add logic here to count fingers based on the angles or distance between start, end, and farthest points.
						fingerCount++;
					}
				}
			}

			return fingerCount;
		}


		private void HandleLeftHandGestures(int fingerCount)
		{
			// Detect gestures based on the number of fingers and set actions
			if (fingerCount == 1)
			{
				_drawColor = Color.Black; // 1 finger - Paint in black
			}
			else if (fingerCount == 0)
			{
				_drawing = false; // Open palm - Stop drawing
			}
			else if (fingerCount == 5)
			{
				_drawing = true; // Back of the hand (open palm but facing opposite)
			}
			else if (fingerCount == 2)
			{
				SaveImage(); // 2 fingers - Save the image
			}
		}

		private void SaveImage()
		{
			string savePath = @"C:\path_to_save\painting.png";
			_canvas.Save(savePath);
			Console.WriteLine($"Image saved to {savePath}");
		}

		private void DrawOnCanvas(Point currentPoint)
		{
			if (_prevPoint.IsEmpty)
			{
				_prevPoint = currentPoint;
			}

			// Draw a line between the previous and current point to simulate brush strokes
			using (Pen pen = new Pen(_drawColor, 5))
			{
				_graphics.DrawLine(pen, _prevPoint, currentPoint);
			}

			_prevPoint = currentPoint; // Update the previous point
		}

		private Point GetRightHandPosition(VectorOfPoint contour)
		{
			// Calculate image moments to find the centroid
			var moments = CvInvoke.Moments(contour);

			int cX = (int)(moments.M10 / moments.M00); // x-coordinate of the centroid
			int cY = (int)(moments.M01 / moments.M00); // y-coordinate of the centroid

			return new Point(cX, cY);  // Return the calculated centroid position
		}


	}
}
