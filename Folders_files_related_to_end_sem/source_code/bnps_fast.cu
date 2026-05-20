#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BLOCK_SIZE 1024
#define CEIL(a, b) ((a - 1) / b + 1)
#define BUFFER 256

#define END 0
#define CON -1
#define VAR -2

// operatorsa
#define LB 1
#define OP_NONE 2
#define OP_AND 3
#define OP_OR 4
#define OP_XOR 5 // NEW: ⊕ XOR operator for BNPS
#define EQ 6
#define NE 7
#define LT 8
#define LE 9
#define GT 10
#define GE 11
#define ADD 12
#define SUB 13
#define MUL 14
#define MOD 15
#define DIV 16
#define EXP 17
#define NEG 18
#define RB 19
#define SQRT_OP 20
#define ABS_OP 21

#define LIMIT (float)0.00001

#define printError(func)                                                       \
  {                                                                            \
    cudaError_t E = func;                                                      \
    if (E != cudaSuccess) {                                                    \
      printf("\nError at line: %d ", __LINE__);                                \
      printf("\nError:  %s ", cudaGetErrorString(E));                          \
    }                                                                          \
  }

typedef struct {
  int type;
  int i, j;
  float value;
} PostfixElement;

typedef struct {
  int proportion;
  int i, j;
} DistFuncElement;

// ─────────────────────────────────────────────────────────────
// Full shunting-yard infix→postfix converter
// Handles: +  -  *  /  unary-minus  ( )
// Operands: $_%d_%d  (variable)  or  numeric literal (int/float)
// ─────────────────────────────────────────────────────────────
static int opPrec(int op) {
  if (op == ADD || op == SUB)
    return 1;
  if (op == MUL || op == DIV)
    return 2;
  if (op == EXP)
    return 3; // exponentiation
  if (op == NEG)
    return 4; // unary minus — highest
  return 0;
}

void infixToPostfix(const char *infix, PostfixElement *postfix,
                    int *postfix_len) {
  // operator stack (stores operator codes)
  int opStack[64];
  int opTop = -1;
  int pos = 0; // write cursor into postfix[]
  int i = 0;   // read cursor into infix[]
  int len = (int)strlen(infix);
  bool prevWasValue = false; // for unary-minus detection

  while (i < len) {
    char c = infix[i];

    // ── skip spaces / parens that enclose the whole expression ──
    if (c == ' ') {
      i++;
      continue;
    }

    // ── function calls: sqrt, abs ────────────────────────────────
    if (c == 's' && i + 3 < len && infix[i + 1] == 'q' && infix[i + 2] == 'r' &&
        infix[i + 3] == 't') {
      opStack[++opTop] = SQRT_OP;
      i += 4; // skip "sqrt"
      prevWasValue = false;
      continue;
    }
    if (c == 'a' && i + 2 < len && infix[i + 1] == 'b' && infix[i + 2] == 's') {
      opStack[++opTop] = ABS_OP;
      i += 3; // skip "abs"
      prevWasValue = false;
      continue;
    }

    // ── left paren ──────────────────────────────────────────────
    if (c == '(') {
      opStack[++opTop] = LB;
      prevWasValue = false;
      i++;
      continue;
    }

    // ── right paren ─────────────────────────────────────────────
    if (c == ')') {
      while (opTop >= 0 && opStack[opTop] != LB) {
        postfix[pos].type = opStack[opTop--];
        pos++;
      }
      if (opTop >= 0)
        opTop--; // pop LB
      // Pop function operator if present (sqrt, abs)
      if (opTop >= 0 &&
          (opStack[opTop] == SQRT_OP || opStack[opTop] == ABS_OP)) {
        postfix[pos].type = opStack[opTop--];
        pos++;
      }
      prevWasValue = true;
      i++;
      continue;
    }

    // ── variable: $_%d_%d ───────────────────────────────────────
    if (c == '$') {
      int a = 0, b = 0;
      i++; // skip $
      i++; // skip _
      while (i < len && infix[i] >= '0' && infix[i] <= '9')
        a = a * 10 + (infix[i++] - '0');
      i++; // skip _
      while (i < len && infix[i] >= '0' && infix[i] <= '9')
        b = b * 10 + (infix[i++] - '0');
      postfix[pos].type = VAR;
      postfix[pos].i = a;
      postfix[pos].j = b;
      postfix[pos].value = 0;
      pos++;
      prevWasValue = true;
      continue;
    }

    // ── numeric literal (integer or float, possibly starting with '.') ─
    if ((c >= '0' && c <= '9') || c == '.') {
      char buf[32];
      int bi = 0;
      while (i < len &&
             ((infix[i] >= '0' && infix[i] <= '9') || infix[i] == '.'))
        buf[bi++] = infix[i++];
      buf[bi] = '\0';
      postfix[pos].type = CON;
      postfix[pos].value = (float)atof(buf);
      postfix[pos].i = 0;
      postfix[pos].j = 0;
      pos++;
      prevWasValue = true;
      continue;
    }

    // ── operators ───────────────────────────────────────────────
    int op = OP_NONE;
    if (c == '+') {
      op = ADD;
      i++;
    } else if (c == '-') {
      // unary minus if nothing pushed yet or previous token was operator/lparen
      if (!prevWasValue)
        op = NEG;
      else
        op = SUB;
      i++;
    } else if (c == '*') {
      op = MUL;
      i++;
    } else if (c == '/') {
      op = DIV;
      i++;
    } else if (c == '^') {
      op = EXP;
      i++;
    } else {
      i++;
      continue;
    } // unknown char — skip

    if (op == NEG) {
      // Right-associative — push directly (higher prec handles itself)
      opStack[++opTop] = NEG;
      prevWasValue = false;
      continue;
    }

    // Left-associative binary op: pop higher-or-equal precedence ops first
    while (opTop >= 0 && opStack[opTop] != LB &&
           opPrec(opStack[opTop]) >= opPrec(op)) {
      postfix[pos].type = opStack[opTop--];
      pos++;
    }
    opStack[++opTop] = op;
    prevWasValue = false;
  }

  // drain remaining operators
  while (opTop >= 0) {
    if (opStack[opTop] != LB) {
      postfix[pos].type = opStack[opTop];
      pos++;
    }
    opTop--;
  }

  postfix[pos].type = END;
  *postfix_len = pos + 1;
}

// ─────────────────────────────────────────────────────────────
// BNPS Kernel
// Changes from gnps_kernel:
//   1. Kernel renamed to bnps_kernel
//   2. XOR (⊕) operator added — parsed from "xor" in predicate string
//      (bnps.py encodes Unicode ⊕ as "xor" when writing input.txt)
// ─────────────────────────────────────────────────────────────
__global__ void
bnps_kernel(char *prodFunction, int *posProd, char *distFunction, int *posDist,
            char *predFunction, int *posPred, int *numberOfVariables,
            float *variables, int steps, int numberOfPrograms,
            PostfixElement *postfix, DistFuncElement *distribution,
            int numberOfMembranes, int *stackOfOps, float *stackPostfixEval,
            float *minVariableInPosFunc, float *valueOfProdFunc,
            float *sumOfProportions, bool *isProgramActive,
            float *variablesTemp, int *progMembrane) {
  int ID = threadIdx.x;
  int total_vars = numberOfVariables[numberOfMembranes];

  for (int step = 0; step < steps; step++) {

    // ── Reset variables[] to snapshot at start of each step ──────────────
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      for (int v = 0; v < total_vars; v++) {
        variables[v] = variablesTemp[v];
      }
    }
    __syncthreads();

    // ── Phase 1: Evaluate Boolean Predicates (parallel, all use snapshot) ─
    // Predicates always use variablesTemp (snapshot) — standard BNPS rule.
    for (int id = threadIdx.x; id < numberOfPrograms; id += blockDim.x) {
      if (progMembrane[id] != blockIdx.x) {
        isProgramActive[id] = false;
        continue;
      }

      int pbeg = posPred[id], pend = posPred[id + 1];

      // No condition → always active
      if (pend - pbeg == 1 && predFunction[pbeg] == '1') {
        isProgramActive[id] = true;
        continue;
      }
      if (pbeg >= pend) {
        isProgramActive[id] = false;
        continue;
      }

      bool result = false;
      bool firstTerm = true;
      int pendingOp = OP_NONE;
      int pos = pbeg;

      while (pos < pend) {

        // ── Handle NOT prefix (¬ encoded as !) ──
        bool negate = false;
        if (predFunction[pos] == '!') {
          negate = true;
          pos++;
        }

        // ── Evaluate one comparison term ──
        bool term = false;

        if (predFunction[pos] == '$') {
          int lhs_a = 0, lhs_b = 0;

          pos++; // $
          pos++; // _
          while (predFunction[pos] >= '0' && predFunction[pos] <= '9')
            lhs_a = lhs_a * 10 + (predFunction[pos++] - '0');
          pos++; // _
          while (predFunction[pos] >= '0' && predFunction[pos] <= '9')
            lhs_b = lhs_b * 10 + (predFunction[pos++] - '0');

          int lhs_idx = numberOfVariables[lhs_b - 1] + (lhs_a - 1);
          float lhs_val = variablesTemp[lhs_idx];

          // Parse comparison operator
          int op = 0;
          if (predFunction[pos] == '>' && predFunction[pos + 1] == '=') {
            op = GE;
            pos += 2;
          } else if (predFunction[pos] == '>') {
            op = GT;
            pos++;
          } else if (predFunction[pos] == '<' && predFunction[pos + 1] == '=') {
            op = LE;
            pos += 2;
          } else if (predFunction[pos] == '<') {
            op = LT;
            pos++;
          } else if (predFunction[pos] == '=' && predFunction[pos + 1] == '=') {
            op = EQ;
            pos += 2;
          } else if (predFunction[pos] == '=') {
            op = EQ;
            pos++;
          } // single = means ==
          else if (predFunction[pos] == '!' && predFunction[pos + 1] == '=') {
            op = NE;
            pos += 2;
          }

          // Parse RHS: may be a variable ($_%d_%d), negative number, or float
          float rhs_val = 0.0f;
          if (predFunction[pos] == '$') {
            // RHS is a variable reference
            int rhs_a = 0, rhs_b = 0;
            pos++; // $
            pos++; // _
            while (pos < pend && predFunction[pos] >= '0' &&
                   predFunction[pos] <= '9')
              rhs_a = rhs_a * 10 + (predFunction[pos++] - '0');
            pos++; // _
            while (pos < pend && predFunction[pos] >= '0' &&
                   predFunction[pos] <= '9')
              rhs_b = rhs_b * 10 + (predFunction[pos++] - '0');
            int rhs_idx = numberOfVariables[rhs_b - 1] + (rhs_a - 1);
            rhs_val = variablesTemp[rhs_idx];
          } else {
            // RHS is a numeric literal (may be negative, may have decimal)
            bool rhs_neg = false;
            if (predFunction[pos] == '-') {
              rhs_neg = true;
              pos++;
            }
            // Integer part
            while (pos < pend && predFunction[pos] >= '0' &&
                   predFunction[pos] <= '9')
              rhs_val = rhs_val * 10 + (predFunction[pos++] - '0');
            // Decimal part (e.g., 0.5)
            if (pos < pend && predFunction[pos] == '.') {
              pos++; // skip '.'
              float frac = 0.1f;
              while (pos < pend && predFunction[pos] >= '0' &&
                     predFunction[pos] <= '9') {
                rhs_val += (predFunction[pos++] - '0') * frac;
                frac *= 0.1f;
              }
            }
            if (rhs_neg)
              rhs_val = -rhs_val;
          }

          switch (op) {
          case LT:
            term = (lhs_val < rhs_val);
            break;
          case LE:
            term = (lhs_val <= rhs_val);
            break;
          case GT:
            term = (lhs_val > rhs_val);
            break;
          case GE:
            term = (lhs_val >= rhs_val);
            break;
          case EQ:
            term = (fabs(lhs_val - rhs_val) < LIMIT);
            break;
          case NE:
            term = (fabs(lhs_val - rhs_val) >= LIMIT);
            break;
          }
        }

        if (negate)
          term = !term;

        // ── Combine with previous term using pending logical op ──
        if (firstTerm) {
          result = term;
          firstTerm = false;
        } else {
          if (pendingOp == OP_AND)
            result = result && term;
          else if (pendingOp == OP_OR)
            result = result || term;
          else if (pendingOp == OP_XOR)
            result = result ^ term; // ⊕ XOR
        }

        // ── Parse next logical connective: and / or / xor ──
        if (predFunction[pos] == 'a' && predFunction[pos + 1] == 'n' &&
            predFunction[pos + 2] == 'd') {
          pendingOp = OP_AND;
          pos += 3;
        } else if (predFunction[pos] == 'o' && predFunction[pos + 1] == 'r') {
          pendingOp = OP_OR;
          pos += 2;
        }
        // ── XOR: bnps.py encodes Unicode ⊕ as the string "xor" ──
        else if (predFunction[pos] == 'x' && predFunction[pos + 1] == 'o' &&
                 predFunction[pos + 2] == 'r') {
          pendingOp = OP_XOR;
          pos += 3;
        } else {
          break; // end of predicate
        }
      }

      isProgramActive[id] = result;
    }
    __syncthreads();

    // ── Phase 2: Sequential within membrane — evaluate then consume ─────
    //
    // Python serial semantics (bnps3.py lines 116-175):
    //   1. snapshot = {all vars}
    //   2. For each membrane m:
    //      a. ALWAYS reset membrane m's vars to snapshot
    //      b. For each active program in m: evaluate then consume
    //         (within-membrane consumption is sequential)
    //   3. After ALL membranes: zero ALL consumed vars globally
    //   4. Distribute onto the zeroed/non-consumed vars
    //
    // KEY INSIGHT: In serial Python, each membrane ALWAYS restores its
    // vars to snapshot regardless of cross-membrane consumption.
    // Cross-membrane consumed vars are only globally zeroed AFTER all
    // membranes finish processing, right before distribution.

    if (threadIdx.x == 0) {
      // Each block owns one membrane: blockIdx.x = membrane index
      int consumed_indices[65536]; // large enough for 800 membranes * many vars
      int num_consumed = 0;

      {
        int m = blockIdx.x; // THIS block handles membrane m
        // ALWAYS reset this membrane's vars to snapshot.
        // Matches serial Python: "for v in membrane.variables:
        //   v.value = snapshot[v.name]; v.wasConsumed = False"
        int mem_var_start = numberOfVariables[m];
        int mem_var_end = numberOfVariables[m + 1];
        for (int v = mem_var_start; v < mem_var_end; v++) {
          variables[v] = variablesTemp[v]; // always restore to snapshot
        }

        for (int id = 0; id < numberOfPrograms; id++) {
          if (progMembrane[id] != m)
            continue;

          valueOfProdFunc[id] = 0.0f;
          if (!isProgramActive[id])
            continue;

          int pbegin = posProd[id];

          // Evaluate production reading from variables[] (working copy).
          {
            float stack[64];
            int top = -1;
            int pos = 0;
            while (postfix[pos + pbegin].type != END) {
              PostfixElement elem = postfix[pos + pbegin];
              if (elem.type == CON) {
                stack[++top] = elem.value;
              } else if (elem.type == VAR) {
                int var_idx = numberOfVariables[elem.j - 1] + (elem.i - 1);
                stack[++top] = variables[var_idx];
              } else if (elem.type == NEG) {
                float a = stack[top--];
                stack[++top] = -a;
              } else if (elem.type == SQRT_OP) {
                float a = stack[top--];
                stack[++top] = sqrtf(fabsf(a));
              } else if (elem.type == ABS_OP) {
                float a = stack[top--];
                stack[++top] = fabsf(a);
              } else {
                float b_val = stack[top--];
                float a_val = stack[top--];
                float result = 0;
                switch (elem.type) {
                case ADD:
                  result = a_val + b_val;
                  break;
                case SUB:
                  result = a_val - b_val;
                  break;
                case MUL:
                  result = a_val * b_val;
                  break;
                case DIV:
                  result = (b_val != 0.0f) ? a_val / b_val : 0.0f;
                  break;
                case EXP:
                  result = powf(a_val, b_val);
                  break;
                }
                stack[++top] = result;
              }
              pos++;
            }
            valueOfProdFunc[id] = (top >= 0) ? stack[top] : 0.0f;
          }

          // Consume: zero all VAR inputs in variables[] (within-membrane
          // sequential semantics). Also record consumed indices for global
          // zeroing after all membranes.
          // Always consume regardless of production value — matches serial
          // Python where wasConsumed is set during evaluate() unconditionally.
          {
            int pos = 0;
            while (postfix[pos + pbegin].type != END) {
              PostfixElement elem = postfix[pos + pbegin];
              if (elem.type == VAR) {
                int var_idx = numberOfVariables[elem.j - 1] + (elem.i - 1);
                variables[var_idx] = 0.0f;
                if (num_consumed < 16384)
                  consumed_indices[num_consumed++] = var_idx;
              }
              pos++;
            }
          }
        }
      }

      // Global zeroing: after ALL membranes finish, zero all consumed vars.
      // Matches serial Python: "for name in consumed_names:
      //   var_lookup[name].value = 0"
      // This undoes any membrane resets that restored consumed vars to
      // snapshot, ensuring distribution adds onto 0 for consumed vars.
      for (int i = 0; i < num_consumed; i++) {
        variables[consumed_indices[i]] = 0.0f;
      }
    }
    __syncthreads();

    // ── Phase 3: Distribute Production Values ────────────────────
    // variables[] now has consumed vars = 0, non-consumed vars = snapshot.
    // Distribution adds (+=) onto these values, matching Python's
    // rule.variable.value += (proportion / total) * newValue
    for (int id = threadIdx.x; id < numberOfPrograms; id += blockDim.x) {
      if (progMembrane[id] != blockIdx.x)
        continue;
      if (!isProgramActive[id])
        continue;

      float prodVal = valueOfProdFunc[id];
      if (prodVal == 0.0f)
        continue;

      int dbegin = posDist[id];
      int pos = 0;

      while (distribution[pos + dbegin].proportion != 0) {
        int tgtA = distribution[pos + dbegin].i - 1;
        int tgtB = distribution[pos + dbegin].j - 1;
        float prop = distribution[pos + dbegin].proportion;

        if (sumOfProportions[id] > 0) {
          float portion = prodVal * (prop / sumOfProportions[id]);
          int idx = numberOfVariables[tgtB] + tgtA;
          atomicAdd(&variables[idx], portion);
        }
        pos++;
      }
    }
    __syncthreads();

    // ── Sync variablesTemp for next step ─────────────────────────
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      for (int v = 0; v < total_vars; v++) {
        variablesTemp[v] = variables[v];
      }
    }
    __syncthreads();
  }
}

// ─────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────
int main(int argc, char **argv) {
  if (argc < 2) {
    printf("BNPS CUDA Simulator\n");
    printf("Usage: %s <input_file>\n", argv[0]);
    printf("  input_file is generated by: python3 bnps.py <model.pep> -p "
           "<steps>\n");
    return 1;
  }

  FILE *ptr = fopen(argv[1], "r");
  if (!ptr) {
    printf("Error: Cannot open file %s\n", argv[1]);
    return 1;
  }

  int numberOfPrograms;
  fscanf(ptr, "%d", &numberOfPrograms);

  int numberOfMembranes;
  fscanf(ptr, "%d", &numberOfMembranes);

  int sizeOfProdFunction;
  fscanf(ptr, "%d", &sizeOfProdFunction);
  char *prodFunction = (char *)malloc((sizeOfProdFunction + 1) * sizeof(char));
  fscanf(ptr, "%s", prodFunction);

  int sizeOfPosProd;
  fscanf(ptr, "%d", &sizeOfPosProd);
  int *posProd = (int *)malloc(sizeOfPosProd * sizeof(int));
  for (int i = 0; i < sizeOfPosProd; i++)
    fscanf(ptr, "%d", &posProd[i]);

  int sizeOfDistFunction;
  fscanf(ptr, "%d", &sizeOfDistFunction);
  char *distFunction = (char *)malloc((sizeOfDistFunction + 1) * sizeof(char));
  fscanf(ptr, "%s", distFunction);

  int sizeOfPosDist;
  fscanf(ptr, "%d", &sizeOfPosDist);
  int *posDist = (int *)malloc(sizeOfPosDist * sizeof(int));
  for (int i = 0; i < sizeOfPosDist; i++)
    fscanf(ptr, "%d", &posDist[i]);

  int sizeOfPredFunction;
  fscanf(ptr, "%d", &sizeOfPredFunction);
  char *predFunction = (char *)malloc((sizeOfPredFunction + 1) * sizeof(char));
  fscanf(ptr, "%s", predFunction);

  int sizeOfPosPred;
  fscanf(ptr, "%d", &sizeOfPosPred);
  int *posPred = (int *)malloc(sizeOfPosPred * sizeof(int));
  for (int i = 0; i < sizeOfPosPred; i++)
    fscanf(ptr, "%d", &posPred[i]);

  int sizeOfNumberOfVariables;
  fscanf(ptr, "%d", &sizeOfNumberOfVariables);
  int *numberOfVariables =
      (int *)malloc((sizeOfNumberOfVariables + 1) * sizeof(int));
  for (int i = 0; i < sizeOfNumberOfVariables; i++)
    fscanf(ptr, "%d", &numberOfVariables[i]);

  int totalVars = 0;
  if (sizeOfNumberOfVariables > numberOfMembranes) {
    totalVars = numberOfVariables[numberOfMembranes];
  } else {
    totalVars = numberOfVariables[sizeOfNumberOfVariables - 1];
  }
  if (sizeOfNumberOfVariables <= numberOfMembranes) {
    numberOfVariables[numberOfMembranes] = totalVars;
  }

  int sizeOfVariables;
  fscanf(ptr, "%d", &sizeOfVariables);
  float *variables = (float *)malloc((totalVars + 1) * sizeof(float));
  for (int i = 0; i < sizeOfVariables && i < totalVars; i++)
    fscanf(ptr, "%f", &variables[i]);
  variables[totalVars] = 1e300;

  int numberOfIterations;
  fscanf(ptr, "%d", &numberOfIterations);

  static char
      membraneNames[2048][BUFFER]; // supports up to 2047 membranes (800+ subs)
  static char varNames[100000]
                      [BUFFER];     // 25.6MB — static keeps it off the stack
  static int varsPerMembrane[2048]; // supports up to 2047 membranes
  memset(varsPerMembrane, 0, sizeof(varsPerMembrane));

  if (numberOfMembranes == 1) {
    varsPerMembrane[0] = totalVars;
  } else {
    // numberOfVariables[] is cumulative: [0, 3, 4, 5] for 3 membranes
    // varsPerMembrane[i] = numberOfVariables[i+1] - numberOfVariables[i]
    for (int i = 0; i < numberOfMembranes; i++)
      varsPerMembrane[i] = numberOfVariables[i + 1] - numberOfVariables[i];
  }

  int varIdx = 0;
  for (int m = 0; m < numberOfMembranes; m++) {
    fscanf(ptr, "%s", membraneNames[m]);
    for (int v = 0; v < varsPerMembrane[m]; v++) {
      if (varIdx < 10000) {
        fscanf(ptr, "%s", varNames[varIdx]);
        varIdx++;
      }
    }
  }

  // Read progMembrane[]: 0-indexed membrane index for each program
  // Written by bnps3.py extractSystem() — exact membrane ownership
  int sizeOfProgMembrane;
  int *progMembrane = (int *)malloc(numberOfPrograms * sizeof(int));
  if (fscanf(ptr, "%d", &sizeOfProgMembrane) == 1 && sizeOfProgMembrane > 0) {
    for (int i = 0; i < sizeOfProgMembrane && i < numberOfPrograms; i++)
      fscanf(ptr, "%d", &progMembrane[i]);
  } else {
    // Fallback: if not present in input.txt, assign all to membrane 0
    for (int i = 0; i < numberOfPrograms; i++)
      progMembrane[i] = 0;
  }
  fclose(ptr);

  // Convert production functions to postfix
  PostfixElement *postfix =
      (PostfixElement *)malloc(sizeOfProdFunction * sizeof(PostfixElement));
  for (int i = 0; i < sizeOfProdFunction; i++)
    postfix[i].type = END;

  for (int prog = 0; prog < numberOfPrograms; prog++) {
    int start = posProd[prog];
    int end = posProd[prog + 1];
    char prodStr[2000];
    int idx = 0;

    // Copy production string, skipping only the outermost wrapping parens
    // added by extractSystem(). Internal parens must be preserved for
    // correct evaluation of expressions like sqrt((a-b)^2 + (c-d)^2).
    int pstart = start;
    int pend = end;
    // Skip leading '(' if present (outer wrapper)
    if (pstart < pend && prodFunction[pstart] == '(')
      pstart++;
    // Find actual end (before trailing ')' if present)
    int actual_end = pend;
    for (int i = pstart; i < pend && prodFunction[i] != '\0'; i++) {
      if (prodFunction[i] == '-' && prodFunction[i + 1] == '>')
        break;
      actual_end = i + 1;
    }
    // Skip trailing ')' if present (outer wrapper)
    if (actual_end > pstart && prodFunction[actual_end - 1] == ')')
      actual_end--;

    for (int i = pstart; i < actual_end && prodFunction[i] != '\0'; i++) {
      if (prodFunction[i] == '-' && i + 1 < actual_end &&
          prodFunction[i + 1] == '>')
        break;
      prodStr[idx++] = prodFunction[i];
    }
    prodStr[idx] = '\0';
    int postfix_len;
    infixToPostfix(prodStr, &postfix[start], &postfix_len);
  }

  // Parse distribution functions
  DistFuncElement *distribution =
      (DistFuncElement *)malloc(sizeOfDistFunction * sizeof(DistFuncElement));
  for (int i = 0; i < sizeOfDistFunction; i++) {
    distribution[i].proportion = 0;
    distribution[i].i = 0;
    distribution[i].j = 0;
  }

  float *sumOfProportions = (float *)malloc(numberOfPrograms * sizeof(float));
  for (int i = 0; i < numberOfPrograms; i++)
    sumOfProportions[i] = 0.0f;

  for (int prog = 0; prog < numberOfPrograms; prog++) {
    int start = posDist[prog];
    int end = posDist[prog + 1];
    char distStr[2000];
    int idx = 0;
    for (int i = start; i < end && distFunction[i] != '\0'; i++)
      distStr[idx++] = distFunction[i];
    distStr[idx] = '\0';

    int distPos = 0, strPos = 0;
    float totalProp = 0.0f;

    while (strPos < (int)strlen(distStr)) {
      // Skip non-digit separators: '+', ' ', etc. between entries
      while (strPos < (int)strlen(distStr) &&
             !(distStr[strPos] >= '0' && distStr[strPos] <= '9')) {
        strPos++;
      }
      if (strPos >= (int)strlen(distStr))
        break;
      int prop = 0;
      while (distStr[strPos] >= '0' && distStr[strPos] <= '9') {
        prop = prop * 10 + (distStr[strPos] - '0');
        strPos++;
      }
      if (prop == 0)
        break;
      strPos++; // skip '|'
      if (strPos < (int)strlen(distStr) && distStr[strPos] == '$') {
        strPos++;
        strPos++; // skip $ _
        int a = 0;
        while (strPos < (int)strlen(distStr) && distStr[strPos] >= '0' &&
               distStr[strPos] <= '9') {
          a = a * 10 + (distStr[strPos] - '0');
          strPos++;
        }
        strPos++; // skip _
        int b = 0;
        while (strPos < (int)strlen(distStr) && distStr[strPos] >= '0' &&
               distStr[strPos] <= '9') {
          b = b * 10 + (distStr[strPos] - '0');
          strPos++;
        }
        distribution[start + distPos].proportion = prop;
        distribution[start + distPos].i = a;
        distribution[start + distPos].j = b;
        totalProp += prop;
        distPos++;
      }
    }
    distribution[start + distPos].proportion = 0;
    sumOfProportions[prog] = totalProp;
  }

  // Allocate device memory
  char *d_prodFunction, *d_distFunction, *d_predFunction;
  int *d_posProd, *d_posDist, *d_posPred, *d_numberOfVariables, *d_stackOfOps;
  float *d_variables, *d_stackPostfixEval, *d_minVariableInPosFunc,
      *d_valueOfProdFunc, *d_sumOfProportions;
  bool *d_isProgramActive;
  PostfixElement *d_postfix;
  DistFuncElement *d_distribution;
  float *d_variablesTemp;
  int *d_progMembrane;

  int memSize = (totalVars + 1) * sizeof(float);
  printError(cudaMalloc((void **)&d_variablesTemp, memSize));
  printError(
      cudaMemcpy(d_variablesTemp, variables, memSize, cudaMemcpyHostToDevice));

  printError(cudaMalloc((void **)&d_prodFunction,
                        (sizeOfProdFunction + 1) * sizeof(char)));
  printError(cudaMalloc((void **)&d_posProd, sizeOfPosProd * sizeof(int)));
  printError(cudaMalloc((void **)&d_distFunction,
                        (sizeOfDistFunction + 1) * sizeof(char)));
  printError(cudaMalloc((void **)&d_posDist, sizeOfPosDist * sizeof(int)));
  printError(cudaMalloc((void **)&d_predFunction,
                        (sizeOfPredFunction + 1) * sizeof(char)));
  printError(cudaMalloc((void **)&d_posPred, sizeOfPosPred * sizeof(int)));
  printError(cudaMalloc((void **)&d_numberOfVariables,
                        (sizeOfNumberOfVariables + 1) * sizeof(int)));
  printError(cudaMalloc((void **)&d_variables, memSize));
  printError(cudaMalloc((void **)&d_postfix,
                        sizeOfProdFunction * sizeof(PostfixElement)));
  printError(cudaMalloc((void **)&d_distribution,
                        sizeOfDistFunction * sizeof(DistFuncElement)));
  printError(
      cudaMalloc((void **)&d_stackOfOps, sizeOfProdFunction * sizeof(int)));
  printError(cudaMalloc((void **)&d_stackPostfixEval,
                        sizeOfProdFunction * sizeof(float)));
  printError(cudaMalloc((void **)&d_minVariableInPosFunc,
                        numberOfPrograms * sizeof(float)));
  printError(cudaMalloc((void **)&d_valueOfProdFunc,
                        numberOfPrograms * sizeof(float)));
  printError(cudaMalloc((void **)&d_sumOfProportions,
                        numberOfPrograms * sizeof(float)));
  printError(
      cudaMalloc((void **)&d_isProgramActive, numberOfPrograms * sizeof(bool)));

  printError(cudaMemcpy(d_prodFunction, prodFunction,
                        (sizeOfProdFunction + 1) * sizeof(char),
                        cudaMemcpyHostToDevice));
  printError(cudaMemcpy(d_posProd, posProd, sizeOfPosProd * sizeof(int),
                        cudaMemcpyHostToDevice));
  printError(cudaMemcpy(d_distFunction, distFunction,
                        (sizeOfDistFunction + 1) * sizeof(char),
                        cudaMemcpyHostToDevice));
  printError(cudaMemcpy(d_posDist, posDist, sizeOfPosDist * sizeof(int),
                        cudaMemcpyHostToDevice));
  printError(cudaMemcpy(d_predFunction, predFunction,
                        (sizeOfPredFunction + 1) * sizeof(char),
                        cudaMemcpyHostToDevice));
  printError(cudaMemcpy(d_posPred, posPred, sizeOfPosPred * sizeof(int),
                        cudaMemcpyHostToDevice));
  printError(cudaMemcpy(d_numberOfVariables, numberOfVariables,
                        (sizeOfNumberOfVariables + 1) * sizeof(int),
                        cudaMemcpyHostToDevice));
  printError(
      cudaMemcpy(d_variables, variables, memSize, cudaMemcpyHostToDevice));
  printError(cudaMemcpy(d_postfix, postfix,
                        sizeOfProdFunction * sizeof(PostfixElement),
                        cudaMemcpyHostToDevice));
  printError(cudaMemcpy(d_distribution, distribution,
                        sizeOfDistFunction * sizeof(DistFuncElement),
                        cudaMemcpyHostToDevice));
  printError(cudaMemcpy(d_sumOfProportions, sumOfProportions,
                        numberOfPrograms * sizeof(float),
                        cudaMemcpyHostToDevice));

  // progMembrane[] was read from input.txt above — exact membrane ownership
  // from bnps3.py extractSystem(), no heuristic needed.
  printError(
      cudaMalloc((void **)&d_progMembrane, numberOfPrograms * sizeof(int)));
  printError(cudaMemcpy(d_progMembrane, progMembrane,
                        numberOfPrograms * sizeof(int),
                        cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  bnps_kernel<<<numberOfMembranes, 32>>>( // multi-block: 1 block per membrane
      d_prodFunction, d_posProd, d_distFunction, d_posDist, d_predFunction,
      d_posPred, d_numberOfVariables, d_variables, numberOfIterations,
      numberOfPrograms, d_postfix, d_distribution, numberOfMembranes,
      d_stackOfOps, d_stackPostfixEval, d_minVariableInPosFunc,
      d_valueOfProdFunc, d_sumOfProportions, d_isProgramActive, d_variablesTemp,
      d_progMembrane);

  cudaDeviceSynchronize();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Time taken: %f ms\n\n", milliseconds);

  printError(
      cudaMemcpy(variables, d_variables, memSize, cudaMemcpyDeviceToHost));

  printf("\n=== Final Results ===\n");
  printf("num_ps = {\n");
  varIdx = 0;
  for (int m = 0; m < numberOfMembranes; m++) {
    printf("  %s:\n", membraneNames[m]);
    printf("    var = {");
    int vars_in_membrane = varsPerMembrane[m];
    for (int v = 0; v < vars_in_membrane; v++) {
      if (v > 0)
        printf(",");
      printf(" %s: %f", varNames[varIdx], variables[varIdx]);
      varIdx++;
    }
    printf(" }\n");
  }
  printf("}\n");

  // Cleanup
  cudaFree(d_variablesTemp);
  cudaFree(d_progMembrane);
  cudaFree(d_prodFunction);
  cudaFree(d_posProd);
  cudaFree(d_distFunction);
  cudaFree(d_posDist);
  cudaFree(d_predFunction);
  cudaFree(d_posPred);
  cudaFree(d_numberOfVariables);
  cudaFree(d_variables);
  cudaFree(d_postfix);
  cudaFree(d_distribution);
  cudaFree(d_stackOfOps);
  cudaFree(d_stackPostfixEval);
  cudaFree(d_minVariableInPosFunc);
  cudaFree(d_valueOfProdFunc);
  cudaFree(d_sumOfProportions);
  cudaFree(d_isProgramActive);

  free(prodFunction);
  free(posProd);
  free(distFunction);
  free(posDist);
  free(predFunction);
  free(posPred);
  free(numberOfVariables);
  free(variables);
  free(postfix);
  free(distribution);
  free(sumOfProportions);
  free(progMembrane);

  return 0;
}
