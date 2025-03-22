"use client"
import { Check, ChevronRight } from "lucide-react"
import { useNavigate } from "react-router-dom"
import { useVerification } from "../../context/VerificationContext"
import { Button } from "./UIComponents"

const VerificationSuccess = () => {
  const navigate = useNavigate()
  const { deviceFingerprint, ipAddress, geoLocation, showAlternativeMethod } = useVerification()

  // Proceed to voting page
  const proceedToVoting = () => {
    // Set a flag in session storage to indicate the user has been verified
    sessionStorage.setItem("voteVerified", "true")

    // Store verification timestamp and method
    const verificationData = {
      timestamp: new Date().toISOString(),
      method: showAlternativeMethod ? "alternative" : "facial",
      deviceFingerprint: deviceFingerprint,
      ipAddress: ipAddress,
      geoLocation: geoLocation,
    }
    sessionStorage.setItem("voteVerificationData", JSON.stringify(verificationData))

    navigate("/voting/cast-vote")
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-col items-center justify-center rounded-lg border border-green-200 bg-green-50 p-6">
        <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-green-100">
          <Check className="h-8 w-8 text-green-600" />
        </div>
        <h3 className="text-lg font-medium text-green-800">Verification Successful</h3>
        <p className="mt-2 text-center text-sm text-green-700">
          Your identity has been verified successfully. You can now proceed to the voting page.
        </p>

        {/* Verification details */}
        <div className="mt-4 w-full rounded-lg border border-green-200 bg-white p-3 text-xs">
          <div className="flex justify-between border-b border-green-100 pb-2">
            <span className="text-gray-600">Verification ID:</span>
            <span className="font-medium text-gray-800">
              {Math.random().toString(36).substring(2, 10).toUpperCase()}
            </span>
          </div>
          <div className="flex justify-between border-b border-green-100 py-2">
            <span className="text-gray-600">Timestamp:</span>
            <span className="font-medium text-gray-800">{new Date().toLocaleString()}</span>
          </div>
          <div className="flex justify-between py-2">
            <span className="text-gray-600">Methods:</span>
            <span className="font-medium text-gray-800">
              {showAlternativeMethod ? "Alternative (Multi-factor)" : "Facial + Document + Biometric"}
            </span>
          </div>
        </div>
      </div>


      <Button onClick={proceedToVoting} className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white">
        Proceed to Voting
        <ChevronRight className="ml-1 h-4 w-4" />
      </Button>
    </div>
  )
}

export default VerificationSuccess

