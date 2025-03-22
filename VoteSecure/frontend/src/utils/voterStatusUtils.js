// Initialize and manage voter status
export const initializeUserVotingStatus = () => {
    // Check if hasVoted is already set
    if (sessionStorage.getItem("hasVoted") === null) {
      // If not set, initialize it to false for new users
      sessionStorage.setItem("hasVoted", "false")
      console.log("Initialized hasVoted status to false for new user")
    }
  }
  
  // Additional voter status utility functions can be added here
  export const checkVoterStatus = () => {
    return sessionStorage.getItem("hasVoted") === "true"
  }
  
  export const setVoterStatus = (hasVoted) => {
    sessionStorage.setItem("hasVoted", hasVoted.toString())
  }
  
  