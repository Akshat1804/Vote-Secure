"use client"

import { createContext, useContext, useState, useEffect } from "react"
import { useNavigate } from "react-router-dom"
import { getCandidates, generateVoteReceipt } from "../services/candidateService"

const VotingContext = createContext()

export const useVoting = () => {
  const context = useContext(VotingContext)
  if (!context) {
    throw new Error("useVoting must be used within a VotingProvider")
  }
  return context
}

export const VotingProvider = ({ children }) => {
  const navigate = useNavigate()
  const [userData, setUserData] = useState(null)
  const [candidates, setCandidates] = useState([])
  const [selectedCandidate, setSelectedCandidate] = useState(null)
  const [isVoting, setIsVoting] = useState(false)
  const [voteSubmitted, setVoteSubmitted] = useState(false)
  const [error, setError] = useState(null)
  const [voteReceipt, setVoteReceipt] = useState(null)
  const [activeTab, setActiveTab] = useState("all")
  const [searchTerm, setSearchTerm] = useState("")
  const [sessionExpired, setSessionExpired] = useState(false)
  const [showSuccessAnimation, setShowSuccessAnimation] = useState(false)
  const [loadingProgress, setLoadingProgress] = useState(0)
  const [showFilters, setShowFilters] = useState(false)

  // Check if user has already voted
  useEffect(() => {
    const hasVoted = sessionStorage.getItem("hasVoted") === "true"
    if (hasVoted) {
      navigate("/dashboard")
    }
  }, [navigate])

  // Load user data
  useEffect(() => {
    const isVerified = sessionStorage.getItem("voteVerified")
    if (!isVerified) {
      navigate("/voting/verify")
      return
    }

    // Load user data
    try {
      const storedUserData = sessionStorage.getItem("userData")
      if (storedUserData) {
        setUserData(JSON.parse(storedUserData))
      }
    } catch (error) {
      console.error("Error loading user data:", error)
    }

    // Load candidates
    loadCandidates()
  }, [navigate])

  // Load candidates
  const loadCandidates = () => {
    const mockCandidates = getCandidates()
    setCandidates(mockCandidates)

    // Simulate loading progress
    let progress = 0
    const interval = setInterval(() => {
      progress += 10
      setLoadingProgress(progress)
      if (progress >= 100) clearInterval(interval)
    }, 100)
  }

  // Filter candidates based on active tab and search term
  const filteredCandidates = candidates.filter((candidate) => {
    const matchesTab = activeTab === "all" || candidate.category === activeTab
    const matchesSearch =
      candidate.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      candidate.party.toLowerCase().includes(searchTerm.toLowerCase())
    return matchesTab && matchesSearch
  })

  // Select a candidate
  const selectCandidate = (candidate) => {
    setSelectedCandidate(candidate)
    // Add subtle haptic feedback if supported
    if (window.navigator && window.navigator.vibrate) {
      window.navigator.vibrate(50)
    }
  }

  // Submit vote
  const submitVote = () => {
    if (!selectedCandidate) {
      setError("Please select a candidate before submitting your vote.")
      setTimeout(() => setError(null), 3000)
      return
    }

    setIsVoting(true)

    // Simulate API call to submit vote
    setTimeout(() => {
      // Generate a vote receipt
      const receipt = generateVoteReceipt(userData)

      setVoteReceipt(receipt)
      setVoteSubmitted(true)
      setIsVoting(false)
      setShowSuccessAnimation(true)

      // Store vote status in session
      sessionStorage.setItem("hasVoted", "true")
      // Store vote timestamp
      sessionStorage.setItem("voteTimestamp", new Date().toISOString())
      // Store receipt in session storage for reference
      sessionStorage.setItem("voteReceipt", JSON.stringify(receipt))

      // Hide success animation after 3 seconds
      setTimeout(() => setShowSuccessAnimation(false), 3000)
    }, 2000)
  }

  // Session expired handler
  const handleSessionExpired = () => {
    setSessionExpired(true)
  }

  // Return to dashboard
  const returnToDashboard = () => {
    navigate("/dashboard")
  }

  // Toggle filters
  const toggleFilters = () => {
    setShowFilters((prev) => !prev)
  }

  return (
    <VotingContext.Provider
      value={{
        userData,
        candidates,
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
      }}
    >
      {children}
    </VotingContext.Provider>
  )
}

