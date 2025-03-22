"use client"

import { useState } from "react"

const VoteReceipt = ({ receipt, onClose }) => {
  const [showQRCode, setShowQRCode] = useState(false)

  return (
    <div className="space-y-6">
      <div className="relative flex flex-col items-center justify-center overflow-hidden rounded-lg border border-purple-200 bg-purple-50 p-6">
        <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-purple-100 shadow-inner">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-8 w-8 text-purple-600"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
            <polyline points="22 4 12 14.01 9 11.01"></polyline>
          </svg>
        </div>
        <h3 className="text-lg font-medium text-purple-800">Vote Recorded Successfully</h3>
        <p className="mt-2 text-center text-sm text-purple-700">
          Your vote has been securely recorded. Thank you for participating in the democratic process.
        </p>

        {/* Vote receipt */}
        {receipt && (
          <div className="mt-4 w-full rounded-lg border border-purple-200 bg-white p-4 text-sm shadow-sm">
            <div className="mb-2 flex items-center justify-between">
              <h4 className="font-medium text-gray-800">Vote Receipt</h4>
              <button
                onClick={() => setShowQRCode(!showQRCode)}
                className="flex items-center rounded-md bg-gray-100 px-2 py-1 text-xs text-gray-600 hover:bg-gray-200"
              >
                {showQRCode ? "Hide QR" : "Show QR"}
              </button>
            </div>

            {showQRCode && (
              <div className="mb-3 flex justify-center">
                <div className="h-32 w-32 rounded-md bg-white p-2 shadow-sm">
                  {/* Placeholder for QR code */}
                  <div className="flex h-full w-full items-center justify-center bg-gray-100 text-xs text-gray-500">
                    QR Code for verification
                  </div>
                </div>
              </div>
            )}

            <div className="space-y-2 text-gray-600">
              <div className="flex justify-between">
                <span>Receipt ID:</span>
                <span className="font-medium">{receipt.id}</span>
              </div>
              <div className="flex justify-between">
                <span>Timestamp:</span>
                <span className="font-medium">{new Date(receipt.timestamp).toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span>Constituency:</span>
                <span className="font-medium">{receipt.constituency}</span>
              </div>
              <div className="flex justify-between">
                <span>Booth ID:</span>
                <span className="font-medium">{receipt.boothId}</span>
              </div>
              <div className="flex justify-between">
                <span>Election ID:</span>
                <span className="font-medium">{receipt.electionId}</span>
              </div>
              <div className="flex justify-between">
                <span>Verification Code:</span>
                <span className="font-medium">{receipt.verificationCode}</span>
              </div>
            </div>
            <div className="mt-3 text-center text-xs text-gray-500">
              You can use this receipt to verify your vote was counted without revealing your choice.
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default VoteReceipt

