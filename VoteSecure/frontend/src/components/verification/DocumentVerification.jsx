import React, { useState } from "react";
import { FileText } from 'lucide-react';
import { useVerification } from "../../context/VerificationContext";
import { Button } from "./UIComponents";

const DocumentVerification = () => {
  const { handleSecondaryVerificationError, setVerificationMethod } = useVerification();
  
  const [documentType, setDocumentType] = useState("aadhaar");
  const [documentNumber, setDocumentNumber] = useState("");
  const [isVerifying, setIsVerifying] = useState(false);
  const [verificationAttempts, setVerificationAttempts] = useState(0);

  const handleSubmit = (e) => {
    e.preventDefault();
    setIsVerifying(true);

    // Simulate verification process
    setTimeout(() => {
      setIsVerifying(false);

      // For demo purposes, accept any input with proper format
      if (
        (documentType === "aadhaar" && /^\d{12}$/.test(documentNumber.replace(/\s/g, ""))) ||
        (documentType === "pan" && /^[A-Z]{5}\d{4}[A-Z]{1}$/.test(documentNumber)) ||
        (documentType === "voter" && /^[A-Z]{3}\d{7}$/.test(documentNumber))
      ) {
        setVerificationMethod("biometric");
      } else {
        setVerificationAttempts(prev => prev + 1);
        if (verificationAttempts >= 2) {
          handleSecondaryVerificationError("Multiple verification failures. Please try the alternative verification method.");
        } else {
          handleSecondaryVerificationError(`Invalid ${documentType.toUpperCase()} number format. Please check and try again.`);
        }
      }
    }, 2000);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="rounded-lg border border-blue-100 bg-blue-50 p-4">
        <div className="flex items-start space-x-3">
          <FileText className="mt-0.5 h-5 w-5 text-blue-600" />
          <div>
            <h4 className="text-sm font-medium text-blue-800">Document Verification</h4>
            <p className="mt-1 text-xs text-blue-700">Please provide your document details for verification</p>
          </div>
        </div>
      </div>

      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700">Document Type</label>
          <select
            value={documentType}
            onChange={(e) => setDocumentType(e.target.value)}
            className="mt-1 block w-full rounded-md border border-gray-300 bg-white px-3 py-2 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
            disabled={isVerifying}
          >
            <option value="aadhaar">Aadhaar Card</option>
            <option value="pan">PAN Card</option>
            <option value="voter">Voter ID</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700">Document Number</label>
          <input
            type="text"
            value={documentNumber}
            onChange={(e) => setDocumentNumber(e.target.value)}
            placeholder={
              documentType === "aadhaar" ? "1234 5678 9012" : documentType === "pan" ? "ABCDE1234F" : "ABC1234567"
            }
            className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
            required
            disabled={isVerifying}
          />
          <p className="mt-1 text-xs text-gray-500">
            {documentType === "aadhaar"
              ? "Enter your 12-digit Aadhaar number"
              : documentType === "pan"
                ? "Enter your 10-character PAN number"
                : "Enter your Voter ID number"}
          </p>
        </div>

        <Button
          type="submit"
          disabled={isVerifying || !documentNumber}
          className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white hover:from-blue-700 hover:to-purple-700"
        >
          {isVerifying ? "Verifying..." : "Verify Document"}
        </Button>
      </div>
    </form>
  );
};

export default DocumentVerification;
