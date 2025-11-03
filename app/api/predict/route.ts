import { type NextRequest, NextResponse } from "next/server"

// No longer depends on external model_data.json file

interface TensePatterns {
  presentKeywords: string[]
  pastKeywords: string[]
  futureKeywords: string[]
}

// Tense-specific patterns extracted from the dataset
const TENSE_PATTERNS: TensePatterns = {
  presentKeywords: [
    // Simple present
    "is",
    "are",
    "am",
    "do",
    "does",
    "have",
    "has",
    // Present continuous
    "is reading",
    "are playing",
    "am working",
    "is studying",
    // Present perfect
    "have been",
    "has been",
    "have made",
    "has made",
    "have created",
    "has created",
    // Tense indicators
    "currently",
    "now",
    "at the moment",
    "always",
    "usually",
    "often",
    "sometimes",
    "never",
    "every day",
    "every week",
    "help",
    "helps",
    "assist",
    "assists",
    "work",
    "works",
    "play",
    "plays",
    "sing",
    "sings",
    "develop",
    "develops",
    "promote",
    "promotes",
    "research",
    "researches",
    "study",
    "studies",
  ],
  pastKeywords: [
    // Simple past
    "was",
    "were",
    "did",
    "had",
    "went",
    "came",
    "made",
    "took",
    "said",
    "told",
    "got",
    "found",
    "left",
    "started",
    "stopped",
    // Specific past tense verbs
    "discovered",
    "created",
    "built",
    "served",
    "caught",
    "bent",
    "repaired",
    "charged",
    "studied",
    "drank",
    "smashed",
    "cooked",
    "attended",
    "slept",
    "practiced",
    "detected",
    "lost",
    "browsed",
    "celebrated",
    "thanked",
    "organized",
    "enjoyed",
    "discussed",
    "competed",
    "lounged",
    "hung",
    // Past continuous
    "was conducting",
    "was sleeping",
    "were sleeping",
    "were browsing",
    "were analyzing",
    "were discussing",
    "had been",
    // Time indicators
    "ago",
    "yesterday",
    "last week",
    "last month",
    "last year",
    "previously",
    "before",
    "earlier",
  ],
  futureKeywords: [
    // Will + verb
    "will",
    "will be",
    "will have",
    "will have been",
    "will be evolving",
    "will be assisting",
    "will be experimenting",
    "will guide",
    "will enhance",
    "will adjust",
    "will encourage",
    "will collaborate",
    "will become",
    "will integrate",
    "will simulate",
    "will connect",
    "will provide",
    "will assist",
    // Going to
    "going to",
    "gonna",
    // Shall
    "shall",
    // Time indicators
    "tomorrow",
    "next week",
    "next month",
    "next year",
    "in the future",
    "by next",
    "by the end of",
    "in a month",
    "over the next",
    "soon",
    "later",
    // Future perfect
    "will have been",
    "will have designed",
    "will have created",
    "will have established",
    "will have been preparing",
  ],
}

class TensePredictor {
  private presentScore = 0
  private pastScore = 0
  private futureScore = 0

  private normalizeText(text: string): string {
    return text
      .toLowerCase()
      .replace(/[^\w\s]/g, "")
      .trim()
  }

  private findPatternMatches(text: string, patterns: string[]): number {
    const normalized = this.normalizeText(text)
    const tokens = normalized.split(/\s+/)
    const bigrams = []
    for (let i = 0; i < tokens.length - 1; i++) {
      bigrams.push(`${tokens[i]} ${tokens[i + 1]}`)
    }

    let matches = 0
    patterns.forEach((pattern) => {
      const patternTokens = pattern.split(/\s+/)

      if (patternTokens.length === 1) {
        // Single word pattern
        if (tokens.includes(pattern)) {
          matches += 2
        }
      } else if (patternTokens.length === 2) {
        // Two-word pattern
        if (bigrams.includes(pattern)) {
          matches += 3 // Bigrams get higher weight
        }
      }
    })

    return matches
  }

  private predictBasedOnPatterns(text: string): number {
    this.presentScore = this.findPatternMatches(text, TENSE_PATTERNS.presentKeywords)
    this.pastScore = this.findPatternMatches(text, TENSE_PATTERNS.pastKeywords)
    this.futureScore = this.findPatternMatches(text, TENSE_PATTERNS.futureKeywords)

    console.log("[v0] Scores - Present:", this.presentScore, "Past:", this.pastScore, "Future:", this.futureScore)

    // Determine prediction
    const maxScore = Math.max(this.presentScore, this.pastScore, this.futureScore)

    if (maxScore === 0) {
      // Default to present if no strong indicators
      return 1
    }

    if (this.presentScore === maxScore) return 1
    if (this.pastScore === maxScore) return 2
    if (this.futureScore === maxScore) return 3

    return 1
  }

  predict(text: string): { tense: string; prediction: number; confidence: number } {
    const prediction = this.predictBasedOnPatterns(text)

    const tenseMapping: Record<number, string> = {
      1: "Present Tense",
      2: "Past Tense",
      3: "Future Tense",
    }

    const totalScore = this.presentScore + this.pastScore + this.futureScore
    let confidence = 0

    if (totalScore > 0) {
      if (prediction === 1) confidence = this.presentScore / totalScore
      else if (prediction === 2) confidence = this.pastScore / totalScore
      else if (prediction === 3) confidence = this.futureScore / totalScore
    } else {
      confidence = 0.33 // Default confidence when no patterns match
    }

    const tense = tenseMapping[prediction] || "Unknown Tense"

    console.log(`[v0] Prediction: ${tense} (confidence: ${(confidence * 100).toFixed(1)}%)`)

    return { tense, prediction, confidence }
  }
}

export async function POST(request: NextRequest) {
  try {
    const { sentence } = await request.json()

    if (!sentence || typeof sentence !== "string") {
      return NextResponse.json({ error: "Invalid input: sentence is required" }, { status: 400 })
    }

    console.log(`[v0] Processing sentence: "${sentence}"`)

    const predictor = new TensePredictor()
    const result = predictor.predict(sentence)

    return NextResponse.json(result)
  } catch (error) {
    console.error("[v0] API Error:", error)
    return NextResponse.json({ error: error instanceof Error ? error.message : "Prediction failed" }, { status: 500 })
  }
}
