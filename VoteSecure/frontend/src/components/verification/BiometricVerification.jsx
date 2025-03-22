import React, { useState, useEffect } from "react";
import { Fingerprint } from 'lucide-react';
import { useVerification } from "../../context/VerificationContext";
import { Progress } from "./UIComponents";

const BiometricVerification = () => {
  const { handleSecondaryVerificationSuccess, handleSecondaryVerificationError, setVerificationMethod } = useVerification();
  
  const [verificationStep, setVerificationStep] = useState(1);
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState("Please follow the instructions");
  const [isVerifying, setIsVerifying] = useState(false);

  useEffect(() => {
    // Simulate biometric verification process
    const steps = [
      { message: "Look straight at the camera", duration: 2000 },
      { message: "Blink slowly", duration: 2000 },
      { message: "Turn your head slightly to the right", duration: 2000 },
      { message: "Turn your head slightly to the left", duration: 2000 },
      { message: "Analyzing biometric data...", duration: 3000 },
    ];

    let currentStep = 0;
    const totalDuration = steps.reduce((sum, step) => sum + step.duration, 0);
    let elapsedTime = 0;

    setIsVerifying(true);
    const interval = setInterval(() => {
      elapsedTime += 100;

      // Update progress
      setProgress((elapsedTime / totalDuration) * 100);

      // Check if we need to move to the next step
      let stepTime = 0;
      for (let i = 0; i <= currentStep; i++) {
        stepTime += steps[i].duration;
      }

      if (elapsedTime >= stepTime && currentStep < steps.length - 1) {
        currentStep++;
        setVerificationStep(currentStep + 1);
        setMessage(steps[currentStep].message);
      }

      // Check if verification is complete
      if (elapsedTime >= totalDuration) {
        clearInterval(interval);
        setIsVerifying(false);

        // 90% chance of success (for demo purposes)
        if (Math.random() < 0.9) {
          setVerificationMethod("signature");
        } else {
          handleSecondaryVerificationError("Biometric verification failed. Please try again.");
        }
      }
    }, 100);

    return () => clearInterval(interval);
  }, [handleSecondaryVerificationSuccess, handleSecondaryVerificationError, setVerificationMethod]);

  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-blue-100 bg-blue-50 p-4">
        <div className="flex items-start space-x-3">
          <Fingerprint className="mt-0.5 h-5 w-5 text-blue-600" />
          <div>
            <h4 className="text-sm font-medium text-blue-800">Biometric Verification</h4>
            <p className="mt-1 text-xs text-blue-700">{message}</p>
          </div>
        </div>
      </div>

      <div className="space-y-2">
        <div className="flex justify-between text-xs text-gray-500">
          <span>Step {verificationStep} of 5</span>
          <span>{Math.round(progress)}%</span>
        </div>
        <Progress value={progress} className="h-2" />
      </div>

      {isVerifying && (
        <div className="flex justify-center">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-blue-600 border-t-transparent"></div>
        </div>
      )}
    </div>
  );
};

export default BiometricVerification;
