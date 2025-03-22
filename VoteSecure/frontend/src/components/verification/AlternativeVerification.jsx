"use client"

import { useState, useEffect, useRef } from "react"
import { Shield, Clock } from "lucide-react"
import { useVerification } from "../../context/VerificationContext"
import { Button, Alert } from "./UIComponents"

const AlternativeVerification = () => {
  const {
    userData,
    handleSecondaryVerificationSuccess,
    handleSecondaryVerificationError,
    handleFailedAttempt,
    formatTime,
    isLockedOut,
    lockoutTime,
  } = useVerification()

  const [step, setStep] = useState(1)
  const [idNumber, setIdNumber] = useState("")
  const [dob, setDob] = useState("")
  const [address, setAddress] = useState("")
  const [phoneOtp, setPhoneOtp] = useState("")
  const [emailOtp, setEmailOtp] = useState("")
  const [enteredPhoneOtp, setEnteredPhoneOtp] = useState("")
  const [enteredEmailOtp, setEnteredEmailOtp] = useState("")
  const [isVerifying, setIsVerifying] = useState(false)
  const [otpSent, setOtpSent] = useState(false)
  const [otpTimer, setOtpTimer] = useState(300) // 5 minutes
  const [otpTimerActive, setOtpTimerActive] = useState(false)
  const [securityQuestion, setSecurityQuestion] = useState("")
  const [securityAnswer, setSecurityAnswer] = useState("")
  const [correctSecurityAnswer, setCorrectSecurityAnswer] = useState("")
  const [securityQuestionVerified, setSecurityQuestionVerified] = useState(false)

  const timerRef = useRef(null)

  // OTP timer effect
  useEffect(() => {
    if (otpTimerActive && otpTimer > 0) {
      timerRef.current = setTimeout(() => {
        setOtpTimer((prev) => prev - 1)
      }, 1000)
    } else if (otpTimer === 0) {
      setOtpSent(false)
      handleSecondaryVerificationError("OTP has expired. Please request a new one.")
      setOtpTimerActive(false)
    }

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current)
    }
  }, [otpTimerActive, otpTimer])

  // Load security question
  useEffect(() => {
    // In a real app, this would be fetched from the server based on the user
    const securityQuestions = [
      { question: "What was the name of your first pet?", answer: "fluffy" },
      { question: "In which city were you born?", answer: "mumbai" },
      { question: "What was the model of your first car?", answer: "swift" },
      { question: "What is your mother's maiden name?", answer: "patel" },
      { question: "What was the name of your primary school?", answer: "st. mary" },
    ]

    const randomIndex = Math.floor(Math.random() * securityQuestions.length)
    setSecurityQuestion(securityQuestions[randomIndex].question)
    setCorrectSecurityAnswer(securityQuestions[randomIndex].answer)
  }, [])

  // Verify personal information
  const verifyPersonalInfo = (e) => {
    e.preventDefault()

    if (isLockedOut) {
      handleSecondaryVerificationError(`Too many failed attempts. Please try again in ${formatTime(lockoutTime)}.`)
      return
    }

    setIsVerifying(true)

    // Simulate verification process
    setTimeout(() => {
      setIsVerifying(false)

      // For demo purposes, accept any input with proper format
      if (idNumber && dob && address) {
        setStep(2)
      } else {
        handleFailedAttempt()
        handleSecondaryVerificationError("Please fill in all required fields correctly.")
      }
    }, 1500)
  }

  // Verify security question
  const verifySecurityQuestion = (e) => {
    e.preventDefault()

    if (isLockedOut) {
      handleSecondaryVerificationError(`Too many failed attempts. Please try again in ${formatTime(lockoutTime)}.`)
      return
    }

    setIsVerifying(true)

    // Simulate verification process
    setTimeout(() => {
      setIsVerifying(false)

      // Case-insensitive comparison
      if (securityAnswer.toLowerCase() === correctSecurityAnswer.toLowerCase()) {
        setSecurityQuestionVerified(true)
        sendOtps()
      } else {
        handleFailedAttempt()
        handleSecondaryVerificationError("Incorrect answer. Please try again.")
      }
    }, 1500)
  }

  // Send OTPs to phone and email
  const sendOtps = () => {
    // Generate 6-digit OTPs
    const phoneOtpCode = Math.floor(100000 + Math.random() * 900000).toString()
    const emailOtpCode = Math.floor(100000 + Math.random() * 900000).toString()

    setPhoneOtp(phoneOtpCode)
    setEmailOtp(emailOtpCode)
    setOtpSent(true)
    setOtpTimer(300) // 5 minutes
    setOtpTimerActive(true)

    // In a real app, these would be sent to the user's phone and email
    console.log(`Phone OTP: ${phoneOtpCode}`)
    console.log(`Email OTP: ${emailOtpCode}`)
  }

  // Verify OTPs
  const verifyOtps = (e) => {
    e.preventDefault()

    if (isLockedOut) {
      handleSecondaryVerificationError(`Too many failed attempts. Please try again in ${formatTime(lockoutTime)}.`)
      return
    }

    setIsVerifying(true)

    // Simulate verification process
    setTimeout(() => {
      setIsVerifying(false)

      if (enteredPhoneOtp === phoneOtp && enteredEmailOtp === emailOtp) {
        setOtpTimerActive(false)
        handleSecondaryVerificationSuccess()
      } else {
        handleFailedAttempt()
        handleSecondaryVerificationError("Incorrect OTP(s). Please check and try again.")
      }
    }, 1500)
  }

  // Resend OTPs
  const resendOtps = () => {
    if (!isLockedOut) {
      sendOtps()
    }
  }

  return (
    <div className="space-y-6">
      <div className="rounded-lg border border-blue-100 bg-blue-50 p-4">
        <div className="flex items-start space-x-3">
          <Shield className="mt-0.5 h-5 w-5 text-blue-600" />
          <div>
            <h4 className="text-sm font-medium text-blue-800">Alternative Verification</h4>
            <p className="mt-1 text-xs text-blue-700">
              {!securityQuestionVerified
                ? "Please verify your identity by answering your security question."
                : otpSent
                  ? "Enter the OTPs sent to your registered phone and email."
                  : "Please verify your identity using your personal information."}
            </p>
          </div>
        </div>
      </div>

      {isLockedOut && (
        <Alert
          variant="warning"
          title="Account Temporarily Locked"
          message={`Too many failed attempts. Please try again in ${formatTime(lockoutTime)}.`}
        />
      )}

      {!securityQuestionVerified ? (
        <form onSubmit={verifySecurityQuestion} className="space-y-4">
          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-700">Security Question</label>
            <div className="rounded-md border border-gray-300 bg-gray-50 px-3 py-2 text-gray-700">
              {securityQuestion}
            </div>
          </div>

          <div className="space-y-2">
            <label htmlFor="securityAnswer" className="block text-sm font-medium text-gray-700">
              Your Answer
            </label>
            <input
              type="text"
              id="securityAnswer"
              value={securityAnswer}
              onChange={(e) => setSecurityAnswer(e.target.value)}
              className="w-full rounded-md border border-gray-300 px-3 py-2 text-gray-900 placeholder-gray-400 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
              placeholder="Enter your answer"
              required
              disabled={isLockedOut || isVerifying}
            />
            <p className="text-xs text-gray-500">For demo purposes, the answer is: {correctSecurityAnswer}</p>
          </div>

          <Button
            type="submit"
            disabled={isVerifying || isLockedOut || !securityAnswer}
            className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white hover:from-blue-700 hover:to-purple-700"
          >
            {isVerifying ? "Verifying..." : "Verify Answer"}
          </Button>
        </form>
      ) : otpSent ? (
        <form onSubmit={verifyOtps} className="space-y-4">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label htmlFor="phoneOtp" className="block text-sm font-medium text-gray-700">
                Phone OTP
              </label>
              <div className="flex items-center text-xs text-gray-500">
                <Clock className="mr-1 h-3 w-3" />
                <span>Expires in: {formatTime(otpTimer)}</span>
              </div>
            </div>
            <input
              type="text"
              id="phoneOtp"
              value={enteredPhoneOtp}
              onChange={(e) => setEnteredPhoneOtp(e.target.value)}
              maxLength={6}
              pattern="\d{6}"
              placeholder="Enter 6-digit OTP sent to your phone"
              className="w-full rounded-md border border-gray-300 px-3 py-2 text-gray-900 placeholder-gray-400 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
              required
              disabled={isLockedOut || isVerifying}
            />
            <p className="text-xs text-gray-500">For demo purposes, the phone OTP is: {phoneOtp}</p>
          </div>

          <div className="space-y-2">
            <label htmlFor="emailOtp" className="block text-sm font-medium text-gray-700">
              Email OTP
            </label>
            <input
              type="text"
              id="emailOtp"
              value={enteredEmailOtp}
              onChange={(e) => setEnteredEmailOtp(e.target.value)}
              maxLength={6}
              pattern="\d{6}"
              placeholder="Enter 6-digit OTP sent to your email"
              className="w-full rounded-md border border-gray-300 px-3 py-2 text-gray-900 placeholder-gray-400 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
              required
              disabled={isLockedOut || isVerifying}
            />
            <p className="text-xs text-gray-500">For demo purposes, the email OTP is: {emailOtp}</p>
          </div>

          <Button
            type="submit"
            disabled={isVerifying || isLockedOut || !enteredPhoneOtp || !enteredEmailOtp}
            className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white hover:from-blue-700 hover:to-purple-700"
          >
            {isVerifying ? "Verifying..." : "Verify OTPs"}
          </Button>

          <div className="text-center text-xs text-gray-500">
            <p>
              Didn't receive the OTPs?{" "}
              <button
                type="button"
                onClick={resendOtps}
                disabled={isLockedOut}
                className="text-blue-600 hover:text-blue-800 hover:underline disabled:opacity-50"
              >
                Resend OTPs
              </button>
            </p>
          </div>
        </form>
      ) : (
        <form onSubmit={verifyPersonalInfo} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700">Voter ID / Aadhaar Number</label>
            <input
              type="text"
              value={idNumber}
              onChange={(e) => setIdNumber(e.target.value)}
              className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
              placeholder="Enter your ID number"
              required
              disabled={isLockedOut || isVerifying}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700">Date of Birth</label>
            <input
              type="date"
              value={dob}
              onChange={(e) => setDob(e.target.value)}
              className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
              required
              disabled={isLockedOut || isVerifying}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700">Registered Address</label>
            <textarea
              value={address}
              onChange={(e) => setAddress(e.target.value)}
              className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
              placeholder="Enter your registered address"
              rows={2}
              required
              disabled={isLockedOut || isVerifying}
            />
          </div>

          <Button
            type="submit"
            disabled={isVerifying || isLockedOut}
            className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white hover:from-blue-700 hover:to-purple-700"
          >
            {isVerifying ? "Verifying..." : "Verify Information"}
          </Button>
        </form>
      )}
    </div>
  )
}

export default AlternativeVerification

