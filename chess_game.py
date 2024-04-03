def pixel_to_chess_coord(x, y, top_left_corner, square_size):

    file = (x - top_left_corner[0]) // square_size
    rank = 8 - ((y - top_left_corner[1]) // square_size)
    return f"{chr(ord('a') + file)}{rank}"


def create_fen_from_positions(piece_positions):

    empty = 0
    fen_rows = []
    for r in range(8, 0, -1):
        fen_row = ""
        for f in range(8):
            pos = f"{chr(ord('a') + f)}{r}"
            if pos in piece_positions:
                if empty > 0:
                    fen_row += str(empty)
                    empty = 0
                fen_row += piece_positions[pos]
            else:
                empty += 1
        if empty > 0:
            fen_row += str(empty)
        fen_rows.append(fen_row)
        empty = 0 
    
    fen = "/".join(fen_rows)
    #fen += f" {active_color} {castling} {en_passant} {halfmove_clock} {fullmove_number}"
    return fen


def fen_to_board(fen):

    board = []
    rows = fen.split('/')
    for row in rows:
        board_row = []
        for char in row:
            if char.isdigit():
                board_row.extend([' '] * int(char)) 
            else:
                board_row.append(char)
        board.append(board_row)
    return board

def compare_positions(fen1, fen2):

    board1 = fen_to_board(fen1)
    board2 = fen_to_board(fen2)
    differences = []
    for i in range(8):
        for j in range(8):
            if board1[i][j] != board2[i][j]:
                differences.append(((chr(j + 97) + str(8 - i)), board1[i][j], board2[i][j]))

    if differences[0][1] == " ":
        return f"{differences[0][0]}{differences[1][0]}"
    elif differences[0][1] != " ":
        return f"{differences[1][0]}{differences[0][0]}"