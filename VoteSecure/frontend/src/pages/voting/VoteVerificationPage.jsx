"use client"

import { Clock, Fingerprint, Shield, Smartphone, User } from "lucide-react"
import { VerificationProvider, useVerification } from "../../context/VerificationContext"
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  CardFooter,
  Progress,
  StepIndicator,
  Alert,
} from "../../components/verification/UIComponents"
import MobileVerification from "../../components/verification/MobileVerification"
import FacialVerification from "../../components/verification/FacialVerification"
import DocumentVerification from "../../components/verification/DocumentVerification"
import BiometricVerification from "../../components/verification/BiometricVerification"
import DigitalSignature from "../../components/verification/DigitalSignature"
import VerificationSuccess from "../../components/verification/VerificationSuccess"
import AlternativeVerification from "../../components/verification/AlternativeVerification"
import { useEffect } from "react"
import { useNavigate } from "react-router-dom"

const VerificationContent = () => {
  const {
    userData,
    step,
    faceVerified,
    secondaryVerificationComplete,
    showAlternativeMethod,
    verificationMethod,
    sessionTimer,
    alertMessage,
    alertVariant,
    formatTime,
    setAlertMessage,
    setError,
  } = useVerification()

  const navigate = useNavigate()

  // Check if user has already voted
  useEffect(() => {
    const hasVoted = sessionStorage.getItem("hasVoted") === "true"
    if (hasVoted) {
      navigate("/dashboard")
    }
  }, [navigate])

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-gradient-to-br from-blue-50 via-white to-purple-50 px-4 py-12 sm:px-6 lg:px-8">
      <div className="w-full max-w-md space-y-6">
        <div
          style={{
            opacity: 1,
            transform: "translateY(0)",
            transition: "opacity 0.5s ease-out, transform 0.5s ease-out",
          }}
        >
          <h1 className="text-center text-3xl font-bold tracking-tight text-gray-900">
            <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              Vote Verification
            </span>
          </h1>
          <p className="mt-2 text-center text-sm text-gray-600">
            Secure verification process for General Elections 2025
          </p>

          {/* User info summary */}
          {userData && (
            <div className="mt-4 flex items-center justify-center">
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-blue-100">
                <User className="h-5 w-5 text-blue-600" />
              </div>
              <div className="ml-3 text-sm">
                <p className="font-medium text-gray-900">{userData.name}</p>
                <p className="text-gray-500">Voter ID: {userData.voterID}</p>
              </div>
            </div>
          )}
        </div>

        <div className="mx-auto w-full max-w-md">
          <StepIndicator steps={["Mobile Verification", "Identity Verification", "Vote"]} currentStep={step} />
        </div>

        {/* Session timer */}
        <div className="flex items-center justify-center text-xs text-gray-500">
          <Clock className="mr-1 h-3 w-3" />
          <span>Session expires in: {formatTime(sessionTimer)}</span>
        </div>

        {/* Alert message */}
        {alertMessage && (
          <Alert
            variant={alertVariant}
            message={alertMessage}
            onClose={() => {
              setAlertMessage(null)
              setError(null)
            }}
          />
        )}

        <div
          style={{
            opacity: 1,
            transform: "translateY(0)",
            transition: "opacity 0.5s ease-out, transform 0.5s ease-out",
          }}
        >
          <Card className="overflow-hidden border-none bg-white/90 shadow-2xl backdrop-blur-sm">
            <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 via-transparent to-purple-500/5 opacity-50" />
            <div className="absolute -top-10 -right-10 h-40 w-40 rounded-full bg-gradient-to-br from-blue-500/10 to-purple-500/10 blur-3xl" />
            <div className="absolute -bottom-10 -left-10 h-40 w-40 rounded-full bg-gradient-to-br from-purple-500/10 to-blue-500/10 blur-3xl" />

            <CardHeader className="relative space-y-1 border-b border-gray-100 pb-4">
              <div className="mb-2 flex items-center justify-between">
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-gradient-to-r from-blue-500 to-purple-600">
                  {step === 1 ? (
                    <Smartphone className="h-5 w-5 text-white" />
                  ) : step === 2 ? (
                    <Fingerprint className="h-5 w-5 text-white" />
                  ) : (
                    <Shield className="h-5 w-5 text-white" />
                  )}
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-sm font-medium text-gray-500">Step {step} of 2</span>
                </div>
              </div>
              <CardTitle className="mt-4 text-xl font-bold">
                {step === 1 ? "Mobile Verification" : "Identity Verification"}
              </CardTitle>
              <CardDescription>
                {step === 1
                  ? "Select the number that was sent to your registered mobile"
                  : "Verify your identity with facial recognition"}
              </CardDescription>
            </CardHeader>

            <CardContent className="relative px-6 pt-6">
              <div className="space-y-6">
                {/* Progress bar */}
                <div className="space-y-2">
                  <div className="flex justify-between text-xs text-gray-500">
                    <span>Mobile Verification</span>
                    <span>Identity Verification</span>
                  </div>
                  <Progress value={step === 1 ? 50 : 100} className="h-2" />
                </div>

                {/* Step 1: Number Verification */}
                {step === 1 && <MobileVerification />}

                {/* Step 2: Identity Verification */}
                {step === 2 && (
                  <div className="space-y-4">
                    {!faceVerified ? (
                      <>
                        {!secondaryVerificationComplete ? (
                          <>
                            {showAlternativeMethod ? (
                              <AlternativeVerification />
                            ) : (
                              <>
                                {verificationMethod === "facial" && <FacialVerification />}
                                {verificationMethod === "document" && <DocumentVerification />}
                                {verificationMethod === "biometric" && <BiometricVerification />}
                                {verificationMethod === "signature" && <DigitalSignature />}
                              </>
                            )}
                          </>
                        ) : null}
                      </>
                    ) : (
                      <VerificationSuccess />
                    )}
                  </div>
                )}
              </div>
            </CardContent>

            <CardFooter className="relative mt-4 flex flex-col space-y-2 border-t border-gray-100 bg-gray-50 px-6 py-4 text-center text-xs text-gray-500">
              <p className="flex items-center justify-center">
                <Shield className="mr-1 h-3 w-3 text-gray-400" />
                Your data is encrypted and securely processed
              </p>
              <p>
                Having trouble?{" "}
                <a href="#" className="text-blue-600 hover:text-blue-800 hover:underline">
                  Contact support
                </a>
              </p>
            </CardFooter>
          </Card>
        </div>

        {/* Add custom CSS for animations */}
        <style
          dangerouslySetInnerHTML={{
            __html: `
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse {
      0% { box-shadow: 0 0 0 0 rgba(79, 70, 229, 0.4); }
      70% { box-shadow: 0 0 0 10px rgba(79, 70, 229, 0); }
      100% { box-shadow: 0 0 0 0 rgba(79, 70, 229, 0); }
    }
    
    .pulse {
      animation: pulse 2s infinite;
    }
  `,
          }}
        />
      </div>
    </div>
  )
}

const VoteVerificationPage = () => {
  return (
    <VerificationProvider>
      <VerificationContent />
    </VerificationProvider>
  )
}

export default VoteVerificationPage

