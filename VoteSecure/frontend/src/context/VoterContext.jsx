"use client"

import { createContext, useState, useContext, useEffect } from "react"

// Create the context
const VoterContext = createContext()

// Create a provider component
export const VoterProvider = ({ children }) => {
  // Initialize state from localStorage if available
  const [voters, setVoters] = useState(() => {
    const savedVoters = localStorage.getItem("voters")
    return savedVoters ? JSON.parse(savedVoters) : []
  })

  const [stats, setStats] = useState({
    totalVoters: 0,
    newRegistrations: 0,
    pendingVerifications: 0,
    verifiedVoters: 0,
    rejectedVoters: 0,
  })

  // Update localStorage when voters change
  useEffect(() => {
    localStorage.setItem("voters", JSON.stringify(voters))

    // Update stats
    const today = new Date().toISOString().split("T")[0]
    const newStats = {
      totalVoters: voters.length,
      newRegistrations: voters.filter((voter) => voter.registrationDate === today).length,
      pendingVerifications: voters.filter((voter) => voter.status === "Pending").length,
      verifiedVoters: voters.filter((voter) => voter.status === "Verified").length,
      rejectedVoters: voters.filter((voter) => voter.status === "Rejected").length,
    }

    setStats(newStats)
  }, [voters])

  // Add a new voter
  const addVoter = (voter) => {
    const newVoter = {
      ...voter,
      id: `V${Math.floor(100000 + Math.random() * 900000)}`,
      registrationDate: new Date().toISOString().split("T")[0],
      status: "Pending",
      // Generate avatar based on initials if no image is provided
      avatar:
        voter.avatar ||
        `https://ui-avatars.com/api/?name=${encodeURIComponent(voter.name)}&background=random&color=fff&size=128`,
    }
    setVoters([...voters, newVoter])
    return newVoter
  }

  // Update a voter
  const updateVoter = (id, updatedData) => {
    const updatedVoters = voters.map((voter) => (voter.id === id ? { ...voter, ...updatedData } : voter))
    setVoters(updatedVoters)
  }

  // Delete a voter
  const deleteVoter = (id) => {
    const filteredVoters = voters.filter((voter) => voter.id !== id)
    setVoters(filteredVoters)
  }

  // Change voter status
  const changeVoterStatus = (id, status) => {
    const updatedVoters = voters.map((voter) => (voter.id === id ? { ...voter, status } : voter))
    setVoters(updatedVoters)
  }

  // Get recent registrations
  const getRecentRegistrations = (limit = 5) => {
    return [...voters].sort((a, b) => new Date(b.registrationDate) - new Date(a.registrationDate)).slice(0, limit)
  }

  return (
    <VoterContext.Provider
      value={{
        voters,
        stats,
        addVoter,
        updateVoter,
        deleteVoter,
        changeVoterStatus,
        getRecentRegistrations,
      }}
    >
      {children}
    </VoterContext.Provider>
  )
}

// Custom hook to use the voter context
export const useVoters = () => {
  const context = useContext(VoterContext)
  if (!context) {
    throw new Error("useVoters must be used within a VoterProvider")
  }
  return context
}

