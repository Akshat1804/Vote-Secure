"use client"

import { useState, useEffect } from "react"
import { useNavigate } from "react-router-dom"
import {
  AlertCircle,
  ChevronRight,
  HelpCircle,
  Info,
  Shield,
  Vote,
  Clock,
  User,
  MapPin,
  ChevronDown,
  Loader2,
  Sparkles,
} from "lucide-react"
import { VotingProvider, useVoting } from "../../context/VotingContext"
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  CardFooter,
  Badge,
  Tooltip,
  Modal,
  ProgressBar,
} from "../../components/voting/VotingUIComponents"
import SessionTimer from "../../components/voting/SessionTimer"
import CandidateCard from "../../components/voting/CandidateCard"
import CandidateDetails from "../../components/voting/CandidateDetails"
import VoteReceipt from "../../components/voting/VoteReceipt"
import { cn } from "../../utils/cn"

const VotingPageContent = () => {
  const navigate = useNavigate()
  const {
    userData,
    filteredCandidates,
    selectedCandidate,
    isVoting,
    voteSubmitted,
    error,
    voteReceipt,
    activeTab,
    searchTerm,
    sessionExpired,
    showSuccessAnimation,
    loadingProgress,
    showFilters,
    setActiveTab,
    setSearchTerm,
    selectCandidate,
    submitVote,
    handleSessionExpired,
    returnToDashboard,
    toggleFilters,
    setError,
  } = useVoting()

  const [showHelp, setShowHelp] = useState(false)
  const [showConfirmation, setShowConfirmation] = useState(false)
  const [showCandidateDetails, setShowCandidateDetails] = useState(false)
  const [selectedCandidateDetails, setSelectedCandidateDetails] = useState(null)
  const [showSecurityInfo, setShowSecurityInfo] = useState(false)
  const [showQRCode, setShowQRCode] = useState(false)

  // Animated elements
  const [animatedElements, setAnimatedElements] = useState({
    header: false,
    candidates: false,
    controls: false,
  })

  // Start animations
  useEffect(() => {
    setTimeout(() => setAnimatedElements((prev) => ({ ...prev, header: true })), 100)
    setTimeout(() => setAnimatedElements((prev) => ({ ...prev, candidates: true })), 300)
    setTimeout(() => setAnimatedElements((prev) => ({ ...prev, controls: true })), 500)
  }, [])

  // View candidate details
  const viewCandidateDetails = (candidate) => {
    setSelectedCandidateDetails(candidate)
    setShowCandidateDetails(true)
  }

  // Handle submit vote button click
  const handleSubmitVote = () => {
    if (!selectedCandidate) {
      setError("Please select a candidate before submitting your vote.")
      setTimeout(() => setError(null), 3000)
      return
    }

    setShowConfirmation(true)
  }

  // Confirm vote
  const confirmVote = () => {
    setShowConfirmation(false)
    submitVote()
  }

  // Toggle help section
  const toggleHelp = () => {
    setShowHelp((prev) => !prev)
  }

  // Toggle security info
  const toggleSecurityInfo = () => {
    setShowSecurityInfo((prev) => !prev)
  }

  // Toggle QR code
  const toggleQRCode = () => {
    setShowQRCode((prev) => !prev)
  }

  // Update the handleSessionExpired function to navigate to the correct verification page
  // const handleSessionExpired = () => {
  //   navigate("/voting/verify")
  // }

  return (
    <div className="flex min-h-screen flex-col bg-gradient-to-br from-indigo-50 via-white to-purple-50 px-4 py-8 sm:px-6 lg:px-8">
      <div className="mx-auto w-full max-w-5xl space-y-6">
        {/* Header */}
        <div
          className={cn(
            "transform transition-all duration-700 ease-out",
            animatedElements.header ? "translate-y-0 opacity-100" : "translate-y-4 opacity-0",
          )}
        >
          <div className="flex flex-col items-center justify-between space-y-4 sm:flex-row sm:space-y-0">
            <div>
              <h1 className="text-center text-3xl font-bold tracking-tight text-gray-900 sm:text-left">
                <span className="bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
                  Cast Your Vote
                </span>
              </h1>
              <p className="mt-2 text-center text-sm text-gray-600 sm:text-left">
                General Elections 2025 - {userData?.constituency || "Your Constituency"}
              </p>
            </div>

            <div className="flex items-center space-x-4">
              <SessionTimer duration={600} onExpire={handleSessionExpired} />

              <button
                onClick={toggleHelp}
                className="rounded-full bg-white p-2 text-gray-500 shadow-sm transition-colors hover:bg-gray-50 hover:text-gray-700"
                aria-label="Help"
              >
                <HelpCircle className="h-5 w-5" />
              </button>
            </div>
          </div>

          {/* User info summary */}
          {userData && (
            <div className="mt-4">
              <div className="overflow-hidden rounded-lg border border-gray-200 bg-white/80 px-4 py-3 shadow-sm backdrop-blur-sm">
                <div className="flex flex-wrap items-center justify-between gap-4">
                  <div className="flex items-center space-x-4">
                    <div className="flex h-12 w-12 items-center justify-center rounded-full bg-gradient-to-r from-indigo-500 to-purple-600">
                      <User className="h-6 w-6 text-white" />
                    </div>
                    <div>
                      <p className="font-medium text-gray-900">{userData.name}</p>
                      <div className="flex items-center space-x-2 text-sm text-gray-500">
                        <span>Voter ID: {userData.voterID}</span>
                        <span>•</span>
                        <Tooltip content="Your registered constituency">
                          <div className="flex items-center">
                            <MapPin className="mr-1 h-3.5 w-3.5" />
                            <span>{userData.constituency || "Central Delhi"}</span>
                          </div>
                        </Tooltip>
                      </div>
                    </div>
                  </div>

                  <Badge variant="indigo" className="animate-pulse">
                    <Clock className="mr-1 h-3.5 w-3.5" />
                    <span>Voting in progress</span>
                  </Badge>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Main content */}
        <div
          className={cn(
            "transform transition-all duration-700 ease-out",
            animatedElements.candidates ? "translate-y-0 opacity-100" : "translate-y-4 opacity-0",
          )}
        >
          <Card className="overflow-hidden border-none bg-white/90 shadow-xl backdrop-blur-sm">
            <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/5 via-transparent to-purple-500/5 opacity-50" />
            <div className="absolute -top-20 -right-20 h-60 w-60 rounded-full bg-gradient-to-br from-indigo-500/10 to-purple-500/10 blur-3xl" />
            <div className="absolute -bottom-20 -left-20 h-60 w-60 rounded-full bg-gradient-to-br from-purple-500/10 to-indigo-500/10 blur-3xl" />

            <CardHeader className="relative space-y-1 border-b border-gray-100 pb-4">
              <div className="mb-2 flex items-center justify-between">
                <div className="flex h-12 w-12 items-center justify-center rounded-full bg-gradient-to-r from-indigo-500 to-purple-600 shadow-md">
                  <Vote className="h-6 w-6 text-white" />
                </div>
                <div className="flex items-center space-x-2">
                  <button
                    onClick={toggleSecurityInfo}
                    className="rounded-full bg-gray-100 p-2 text-gray-500 transition-colors hover:bg-gray-200 hover:text-gray-700"
                    aria-label="Security Information"
                  >
                    <Shield className="h-4 w-4" />
                  </button>
                  <button
                    onClick={toggleHelp}
                    className="rounded-full bg-gray-100 p-2 text-gray-500 transition-colors hover:bg-gray-200 hover:text-gray-700"
                    aria-label="Help"
                  >
                    <HelpCircle className="h-4 w-4" />
                  </button>
                </div>
              </div>
              <CardTitle className="mt-4 text-2xl font-bold">
                {voteSubmitted ? "Vote Submitted Successfully" : "Select Your Candidate"}
              </CardTitle>
              <CardDescription>
                {voteSubmitted
                  ? "Thank you for participating in the democratic process"
                  : "Choose one candidate from the list below"}
              </CardDescription>
            </CardHeader>

            <CardContent className="relative px-6 pt-6">
              <div className="space-y-6">
                {/* Help section */}
                {showHelp && (
                  <div className="rounded-lg border border-indigo-100 bg-indigo-50 p-4 text-sm">
                    <h4 className="mb-2 flex items-center font-medium text-indigo-800">
                      <HelpCircle className="mr-2 h-4 w-4" />
                      Voting Instructions
                    </h4>
                    <div className="space-y-2 text-indigo-700">
                      <p>• Review each candidate's information carefully before making your choice.</p>
                      <p>• Click on a candidate card to select them.</p>
                      <p>• Click "View Details" to see more information about a candidate.</p>
                      <p>• After selecting, click "Submit Vote" to confirm your choice.</p>
                      <p>• You will have one final chance to confirm before your vote is recorded.</p>
                      <p>• Once submitted, your vote cannot be changed.</p>
                    </div>
                    <button onClick={toggleHelp} className="mt-2 text-indigo-600 hover:text-indigo-800 hover:underline">
                      Close Help
                    </button>
                  </div>
                )}

                {error && (
                  <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-center text-sm text-red-700">
                    <AlertCircle className="mr-1 inline-block h-4 w-4" />
                    {error}
                  </div>
                )}

                {!voteSubmitted ? (
                  <>
                    {/* Search and filter */}
                    <div className="mb-4 space-y-3">
                      <div className="flex flex-wrap items-center gap-2">
                        <div className="relative flex-1">
                          <input
                            type="text"
                            placeholder="Search candidates or parties..."
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                            className="w-full rounded-md border border-gray-300 bg-white px-4 py-2 pl-10 text-sm shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                          />
                          <div className="absolute inset-y-0 left-0 flex items-center pl-3 text-gray-400">
                            <svg
                              xmlns="http://www.w3.org/2000/svg"
                              className="h-4 w-4"
                              fill="none"
                              viewBox="0 0 24 24"
                              stroke="currentColor"
                            >
                              <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                              />
                            </svg>
                          </div>
                        </div>

                        <button
                          onClick={toggleFilters}
                          className="flex items-center rounded-md border border-gray-300 bg-white px-3 py-2 text-sm font-medium text-gray-700 shadow-sm hover:bg-gray-50"
                        >
                          <span>Filters</span>
                          <ChevronDown
                            className={`ml-1 h-4 w-4 transition-transform ${showFilters ? "rotate-180" : ""}`}
                          />
                        </button>
                      </div>

                      {showFilters && (
                        <div className="rounded-md border border-gray-200 bg-gray-50 p-3">
                          <div className="flex flex-wrap gap-2">
                            <button
                              onClick={() => setActiveTab("all")}
                              className={`rounded-full px-3 py-1 text-xs font-medium transition-colors ${
                                activeTab === "all"
                                  ? "bg-indigo-100 text-indigo-800"
                                  : "bg-white text-gray-700 hover:bg-gray-100"
                              }`}
                            >
                              All Candidates
                            </button>
                            <button
                              onClick={() => setActiveTab("national")}
                              className={`rounded-full px-3 py-1 text-xs font-medium transition-colors ${
                                activeTab === "national"
                                  ? "bg-orange-100 text-orange-800"
                                  : "bg-white text-gray-700 hover:bg-gray-100"
                              }`}
                            >
                              National Parties
                            </button>
                            <button
                              onClick={() => setActiveTab("regional")}
                              className={`rounded-full px-3 py-1 text-xs font-medium transition-colors ${
                                activeTab === "regional"
                                  ? "bg-green-100 text-green-800"
                                  : "bg-white text-gray-700 hover:bg-gray-100"
                              }`}
                            >
                              Regional Parties
                            </button>
                            <button
                              onClick={() => setActiveTab("independent")}
                              className={`rounded-full px-3 py-1 text-xs font-medium transition-colors ${
                                activeTab === "independent"
                                  ? "bg-purple-100 text-purple-800"
                                  : "bg-white text-gray-700 hover:bg-gray-100"
                              }`}
                            >
                              Independents
                            </button>
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Candidates list */}
                    {loadingProgress < 100 ? (
                      <div className="flex flex-col items-center justify-center py-8">
                        <Loader2 className="mb-2 h-8 w-8 animate-spin text-indigo-500" />
                        <p className="text-sm text-gray-500">Loading candidates...</p>
                        <div className="mt-4 w-64">
                          <ProgressBar value={loadingProgress} max={100} />
                        </div>
                      </div>
                    ) : filteredCandidates.length === 0 ? (
                      <div className="flex flex-col items-center justify-center rounded-lg border border-gray-200 bg-gray-50 py-8">
                        <div className="mb-2 rounded-full bg-gray-100 p-3">
                          <AlertCircle className="h-6 w-6 text-gray-400" />
                        </div>
                        <h3 className="text-lg font-medium text-gray-700">No candidates found</h3>
                        <p className="mt-1 text-sm text-gray-500">Try adjusting your search or filters</p>
                      </div>
                    ) : (
                      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                        {filteredCandidates.map((candidate) => (
                          <CandidateCard
                            key={candidate.id}
                            candidate={candidate}
                            isSelected={selectedCandidate?.id === candidate.id}
                            onSelect={selectCandidate}
                            onViewDetails={viewCandidateDetails}
                          />
                        ))}
                      </div>
                    )}

                    {/* Security information */}
                    <button
                      onClick={toggleSecurityInfo}
                      className="flex w-full items-center justify-between rounded-lg border border-gray-200 bg-gray-50 px-4 py-2 text-left text-sm text-gray-700 transition-colors hover:bg-gray-100"
                    >
                      <div className="flex items-center">
                        <Shield className="mr-2 h-4 w-4 text-gray-500" />
                        <span>Voting Security Information</span>
                      </div>
                      <ChevronRight
                        className={`h-4 w-4 text-gray-500 transition-transform ${showSecurityInfo ? "rotate-90" : ""}`}
                      />
                    </button>

                    {showSecurityInfo && (
                      <div className="rounded-lg border border-gray-200 bg-gray-50 p-4 text-xs text-gray-600">
                        <h5 className="mb-2 font-medium text-gray-700">How We Protect Your Vote</h5>
                        <ul className="space-y-1">
                          <li>• Your vote is encrypted and anonymized</li>
                          <li>• No connection between your identity and your vote is stored</li>
                          <li>• The system uses blockchain technology to prevent tampering</li>
                          <li>
                            • You will receive a verification receipt that you can use to verify your vote was counted
                          </li>
                          <li>• Independent auditors verify the integrity of the voting system</li>
                        </ul>
                      </div>
                    )}

                    {/* Submit button */}
                    <div className="flex justify-center pt-4">
                      <button
                        onClick={handleSubmitVote}
                        disabled={!selectedCandidate || isVoting}
                        className="group relative w-full overflow-hidden rounded-md bg-gradient-to-r from-indigo-600 to-purple-600 px-8 py-3 text-white transition-all hover:from-indigo-700 hover:to-purple-700 disabled:opacity-50 sm:w-auto"
                      >
                        <span className="relative z-10 flex items-center justify-center">
                          {isVoting ? (
                            <>
                              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                              Processing...
                            </>
                          ) : (
                            <>Submit Vote</>
                          )}
                        </span>
                        <span className="absolute inset-0 -translate-y-full bg-gradient-to-r from-indigo-700 to-purple-700 transition-transform duration-300 ease-out group-hover:translate-y-0"></span>
                      </button>
                    </div>
                  </>
                ) : (
                  // Vote submitted success
                  <VoteReceipt receipt={voteReceipt} onClose={returnToDashboard} />
                )}
              </div>
            </CardContent>

            <CardFooter className="relative mt-4 flex flex-col space-y-2 border-t border-gray-100 bg-gray-50 px-6 py-4 text-center text-xs text-gray-500">
              <p className="flex items-center justify-center">
                <Shield className="mr-1 h-3 w-3 text-gray-400" />
                Your vote is confidential and securely processed
              </p>
              <p>
                Having trouble?{" "}
                <a href="#" className="text-indigo-600 hover:text-indigo-800 hover:underline">
                  Contact election support
                </a>
              </p>
            </CardFooter>
          </Card>
        </div>

        {/* Footer */}
        <div
          className={cn(
            "transform transition-all duration-700 ease-out",
            animatedElements.controls ? "translate-y-0 opacity-100" : "translate-y-4 opacity-0",
          )}
        >
          <div className="rounded-lg border border-gray-200 bg-white/80 p-4 text-center text-sm text-gray-600 backdrop-blur-sm">
            <p className="flex items-center justify-center">
              <Sparkles className="mr-2 h-4 w-4 text-indigo-500" />
              <span>Your vote matters! Thank you for participating in the democratic process.</span>
            </p>
          </div>
        </div>
      </div>

      {/* Candidate details modal */}
      <Modal isOpen={showCandidateDetails} onClose={() => setShowCandidateDetails(false)} title="Candidate Details">
        <CandidateDetails
          candidate={selectedCandidateDetails}
          onClose={() => setShowCandidateDetails(false)}
          onSelect={selectCandidate}
        />
      </Modal>

      {/* Confirmation modal */}
      <Modal isOpen={showConfirmation} onClose={() => setShowConfirmation(false)} title="Confirm Your Vote">
        {selectedCandidate && (
          <div className="space-y-4">
            <div className="rounded-lg border border-yellow-200 bg-yellow-50 p-3 text-sm text-yellow-800">
              <div className="flex items-start">
                <Info className="mr-2 h-5 w-5 flex-shrink-0 text-yellow-600" />
                <p>
                  You are about to cast your vote for <strong>{selectedCandidate.name}</strong> of{" "}
                  <strong>{selectedCandidate.party}</strong>. This action cannot be undone.
                </p>
              </div>
            </div>

            <div className="flex items-center space-x-4 rounded-lg border border-gray-200 bg-white p-4">
              <div className="text-3xl">{selectedCandidate.symbol}</div>
              <div>
                <h3 className="font-medium">{selectedCandidate.name}</h3>
                <p className="text-sm text-gray-600">{selectedCandidate.party}</p>
              </div>
            </div>

            <p className="text-center text-sm text-gray-600">
              Please confirm that this is your intended choice. Once submitted, your vote cannot be changed.
            </p>

            <div className="flex justify-center space-x-3 pt-2">
              <button
                onClick={() => setShowConfirmation(false)}
                className="rounded-md border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 transition-colors hover:bg-gray-50"
              >
                Go Back
              </button>
              <button
                onClick={confirmVote}
                className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-indigo-700"
              >
                Confirm Vote
              </button>
            </div>
          </div>
        )}
      </Modal>

      {/* Session expired modal */}
      <Modal isOpen={sessionExpired} onClose={() => {}} title="Session Expired">
        <div className="space-y-4">
          <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-800">
            <div className="flex items-start">
              <AlertCircle className="mr-2 h-5 w-5 flex-shrink-0 text-red-600" />
              <p>
                Your voting session has expired due to inactivity. For security reasons, you will need to verify your
                identity again.
              </p>
            </div>
          </div>

          <div className="flex justify-center pt-2">
            <button
              onClick={() => navigate("/voting/verify")}
              className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700"
            >
              Verify Identity Again
            </button>
          </div>
        </div>
      </Modal>

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
          
          @keyframes spin-slow {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
          }
          
          .animate-spin-slow {
            animation: spin-slow 10s linear infinite;
          }
        `,
        }}
      />
    </div>
  )
}

// Wrap the component with the VotingProvider
const VotingPage = () => {
  return (
    <VotingProvider>
      <VotingPageContent />
    </VotingProvider>
  )
}

export default VotingPage

