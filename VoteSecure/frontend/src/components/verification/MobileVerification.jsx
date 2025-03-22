import React from "react";
import { AlertCircle, ChevronRight, Shield, Smartphone } from 'lucide-react';
import { useVerification } from "../../context/VerificationContext";
import { Button, Alert } from "./UIComponents";

const MobileVerification = () => {
  const {
    userData,
    verificationNumbers,
    error,
    alertMessage,
    showSecurityInfo,
    isLockedOut,
    verifyNumber,
    generateVerificationNumbers,
    toggleSecurityInfo,
  } = useVerification();

  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-blue-100 bg-blue-50 p-4">
        <div className="flex items-start space-x-3">
          <Smartphone className="mt-0.5 h-5 w-5 text-blue-600" />
          <div>
            <h4 className="text-sm font-medium text-blue-800">Verification Code</h4>
            <p className="mt-1 text-xs text-blue-700">
              A verification code has been sent to your registered mobile number ending in
              <span className="font-medium">
                {" "}
                {userData?.phoneNumber ? userData.phoneNumber.slice(-4) : "****"}
              </span>
              . Select the correct number below.
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4">
        {verificationNumbers.map((number, index) => (
          <button
            key={index}
            onClick={() => verifyNumber(number)}
            className="group relative flex h-16 items-center justify-center overflow-hidden rounded-lg border border-gray-200 bg-white text-xl font-semibold text-gray-800 shadow-sm transition-all hover:border-blue-300 hover:bg-blue-50 hover:shadow"
            disabled={isLockedOut}
          >
            <div className="absolute inset-0 bg-gradient-to-br from-blue-50 to-purple-50 opacity-0 transition-opacity group-hover:opacity-100"></div>
            <span className="relative">{number}</span>
          </button>
        ))}
      </div>

      {error && !alertMessage && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-center text-sm text-red-700">
          <AlertCircle className="mr-1 inline-block h-4 w-4" />
          {error}
        </div>
      )}

      <div className="text-center text-xs text-gray-500">
        <p>
          Didn't receive the code?{" "}
          <button
            onClick={generateVerificationNumbers}
            className="text-blue-600 hover:text-blue-800 hover:underline"
            disabled={isLockedOut}
          >
            Resend Code
          </button>
        </p>
      </div>

      {/* Security information */}
      <button
        onClick={toggleSecurityInfo}
        className="flex w-full items-center justify-between rounded-lg border border-gray-200 bg-gray-50 px-4 py-2 text-left text-sm text-gray-700 hover:bg-gray-100"
      >
        <div className="flex items-center">
          <Shield className="mr-2 h-4 w-4 text-gray-500" />
          <span>Security Information</span>
        </div>
        <ChevronRight
          className={`h-4 w-4 text-gray-500 transition-transform ${showSecurityInfo ? "rotate-90" : ""}`}
        />
      </button>

      {showSecurityInfo && (
        <div className="rounded-lg border border-gray-200 bg-gray-50 p-4 text-xs text-gray-600">
          <h5 className="mb-2 font-medium text-gray-700">How We Protect Your Vote</h5>
          <ul className="space-y-1">
            <li>• All data is encrypted using industry-standard protocols</li>
            <li>• Multi-factor authentication ensures only you can cast your vote</li>
            <li>• Your vote is anonymized after verification</li>
            <li>• Our system is regularly audited by independent security experts</li>
            <li>• No personal data is stored with your voting record</li>
            <li>• Anti-fraud measures detect and prevent multiple voting attempts</li>
            <li>• Continuous monitoring for suspicious activities</li>
          </ul>
        </div>
      )}
    </div>
  );
};

export default MobileVerification;
