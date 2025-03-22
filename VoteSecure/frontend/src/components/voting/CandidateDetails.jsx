"use client"

const CandidateDetails = ({ candidate, onClose, onSelect }) => {
  if (!candidate) return null

  return (
    <div className="space-y-4">
      <div className="flex items-center space-x-4">
        <div className="relative h-20 w-20 overflow-hidden rounded-full border-2 border-white shadow-md">
          <img
            src={candidate.image || "/placeholder.svg"}
            alt={candidate.name}
            className="h-full w-full object-cover"
          />
        </div>
        <div>
          <h3 className="text-xl font-medium">{candidate.name}</h3>
          <p className="text-gray-600">{candidate.party}</p>
          <div className="mt-1 text-3xl">{candidate.symbol}</div>
        </div>
      </div>

      <div className="space-y-3 rounded-lg border border-gray-200 bg-gray-50 p-4">
        <div>
          <span className="font-medium text-gray-700">Age: </span>
          <span>{candidate.details.age} years</span>
        </div>
        <div>
          <span className="font-medium text-gray-700">Education: </span>
          <span>{candidate.details.education}</span>
        </div>
        <div>
          <span className="font-medium text-gray-700">Experience: </span>
          <span>{candidate.details.experience}</span>
        </div>
      </div>

      <div>
        <h4 className="mb-2 font-medium text-gray-800">Key Achievements:</h4>
        <ul className="list-inside list-disc space-y-1 rounded-lg border border-gray-200 bg-white p-3 text-sm">
          {candidate.details.achievements.map((achievement, index) => (
            <li key={index}>{achievement}</li>
          ))}
        </ul>
      </div>

      <div>
        <h4 className="mb-2 font-medium text-gray-800">Key Manifesto Points:</h4>
        <ul className="list-inside list-disc space-y-1 rounded-lg border border-gray-200 bg-white p-3 text-sm">
          {candidate.details.manifesto.map((point, index) => (
            <li key={index}>{point}</li>
          ))}
        </ul>
      </div>

      <div className="flex justify-end space-x-3 pt-2">
        <button
          onClick={onClose}
          className="rounded-md border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 transition-colors hover:bg-gray-50"
        >
          Close
        </button>
        <button
          onClick={() => {
            onSelect(candidate)
            onClose()
          }}
          className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-indigo-700"
        >
          Select Candidate
        </button>
      </div>
    </div>
  )
}

export default CandidateDetails

