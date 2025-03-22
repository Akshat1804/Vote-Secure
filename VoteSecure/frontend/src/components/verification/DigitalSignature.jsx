import React, { useState, useEffect, useRef } from "react";
import { Key } from 'lucide-react';
import { useVerification } from "../../context/VerificationContext";
import { Button } from "./UIComponents";

const DigitalSignature = () => {
  const { handleSecondaryVerificationSuccess, handleSecondaryVerificationError } = useVerification();
  
  const [signature, setSignature] = useState("");
  const [isVerifying, setIsVerifying] = useState(false);
  const [signatureQuality, setSignatureQuality] = useState(0);

  const handleSubmit = (e) => {
    e.preventDefault();

    if (!signature) {
      handleSecondaryVerificationError("Please draw your signature");
      return;
    }

    if (signatureQuality < 30) {
      handleSecondaryVerificationError("Signature quality is too low. Please draw a clearer signature.");
      return;
    }

    setIsVerifying(true);

    // Simulate verification process
    setTimeout(() => {
      setIsVerifying(false);
      handleSecondaryVerificationSuccess();
    }, 2000);
  };

  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [points, setPoints] = useState([]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#1e40af";
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    const startDrawing = (e) => {
      setIsDrawing(true);
      setPoints([]);
      ctx.beginPath();
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      ctx.moveTo(x, y);
      setPoints(prev => [...prev, { x, y }]);
    };

    const draw = (e) => {
      if (!isDrawing) return;

      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      ctx.lineTo(x, y);
      ctx.stroke();
      setPoints(prev => [...prev, { x, y }]);

      // Calculate signature quality based on number of points and coverage
      const pointCount = points.length;
      const canvasArea = canvas.width * canvas.height;
      
      // Calculate bounding box of signature
      if (pointCount > 10) {
        const xCoords = points.map(p => p.x);
        const yCoords = points.map(p => p.y);
        const minX = Math.min(...xCoords);
        const maxX = Math.max(...xCoords);
        const minY = Math.min(...yCoords);
        const maxY = Math.max(...yCoords);
        
        const signatureArea = (maxX - minX) * (maxY - minY);
        const coverage = signatureArea / canvasArea;
        
        // Quality is based on point count and coverage
        const newQuality = Math.min(100, Math.max(0, pointCount / 5 + coverage * 100));
        setSignatureQuality(newQuality);
      }
    };

    const stopDrawing = () => {
      if (isDrawing) {
        setIsDrawing(false);
        setSignature(canvas.toDataURL());
      }
    };

    canvas.addEventListener("mousedown", startDrawing);
    canvas.addEventListener("mousemove", draw);
    canvas.addEventListener("mouseup", stopDrawing);
    canvas.addEventListener("mouseout", stopDrawing);

    // Touch events
    canvas.addEventListener("touchstart", (e) => {
      e.preventDefault();
      const touch = e.touches[0];
      const mouseEvent = new MouseEvent("mousedown", {
        clientX: touch.clientX,
        clientY: touch.clientY,
      });
      canvas.dispatchEvent(mouseEvent);
    });

    canvas.addEventListener("touchmove", (e) => {
      e.preventDefault();
      const touch = e.touches[0];
      const mouseEvent = new MouseEvent("mousemove", {
        clientX: touch.clientX,
        clientY: touch.clientY,
      });
      canvas.dispatchEvent(mouseEvent);
    });

    canvas.addEventListener("touchend", (e) => {
      e.preventDefault();
      const mouseEvent = new MouseEvent("mouseup", {});
      canvas.dispatchEvent(mouseEvent);
    });

    return () => {
      canvas.removeEventListener("mousedown", startDrawing);
      canvas.removeEventListener("mousemove", draw);
      canvas.removeEventListener("mouseup", stopDrawing);
      canvas.removeEventListener("mouseout", stopDrawing);

      canvas.removeEventListener("touchstart", startDrawing);
      canvas.removeEventListener("touchmove", draw);
      canvas.removeEventListener("touchend", stopDrawing);
    };
  }, [isDrawing, points]);

  const clearSignature = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    setSignature("");
    setPoints([]);
    setSignatureQuality(0);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="rounded-lg border border-blue-100 bg-blue-50 p-4">
        <div className="flex items-start space-x-3">
          <Key className="mt-0.5 h-5 w-5 text-blue-600" />
          <div>
            <h4 className="text-sm font-medium text-blue-800">Digital Signature</h4>
            <p className="mt-1 text-xs text-blue-700">Please draw your signature below to verify your identity</p>
          </div>
        </div>
      </div>

      <div className="space-y-4">
        <div className="rounded-lg border border-gray-300 bg-white p-2">
          <canvas
            ref={canvasRef}
            width={400}
            height={150}
            className="w-full cursor-crosshair rounded border border-gray-200 bg-gray-50"
          />
          <div className="mt-2 flex justify-between items-center">
            <p className="text-xs text-gray-500">Draw your signature above</p>
            <div className="flex items-center space-x-2">
              <div className="h-1.5 w-16 rounded-full bg-gray-200">
                <div
                  className={`h-full rounded-full transition-all duration-300 ${
                    signatureQuality > 70
                      ? "bg-green-500"
                      : signatureQuality > 40
                        ? "bg-yellow-500"
                        : "bg-red-500"
                  }`}
                  style={{ width: `${signatureQuality}%` }}
                ></div>
              </div>
              <button
                type="button"
                onClick={clearSignature}
                className="text-xs text-blue-600 hover:text-blue-800 hover:underline"
              >
                Clear
              </button>
            </div>
          </div>
        </div>

        <Button
          type="submit"
          disabled={isVerifying || !signature || signatureQuality < 30}
          className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white hover:from-blue-700 hover:to-purple-700"
        >
          {isVerifying ? "Verifying..." : "Submit Signature"}
        </Button>
      </div>
    </form>
  );
};

export default DigitalSignature;
