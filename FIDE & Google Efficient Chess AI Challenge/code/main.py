import time
import os
import json
from integration import stub

class ChessGameManager:
    def __init__(self):
        self.env = stub.Environment.setup()
        self.agent_states = None
    
    def _display_board(self, board, score):
        print("--- 현재 보드 상태 ---")
        for row in board:
            # 리스트 안의 요소를 공백으로 구분하여 한 줄씩 출력
            print(" ".join(row))
        print(f"Material Score: {score}")
        print("----------------------")

    @staticmethod
    def _clear_screen():
        os.system('cls' if os.name == 'nt' else 'clear')

    def _get_player_info(self, state):
        idx = state.observation.get('currentPlayer', 0)
        name = "백(White)" if idx == 1 else "흑(Black)"
        return idx, name

    def _save_result(self):
        with open("game_result.json", "w") as f:
            json.dump(self.env.toJSON(), f, indent=4)
        print("결과가 game_result.json으로 저장되었습니다.")

    def play(self):
        self.env.reset()
        self.agent_states = self.env.step([None, None])
        
        print("게임 시작!")

        while not self.env.done:
            self._clear_screen()
            
            current_idx, player_name = self._get_player_info(self.agent_states[0])
            obs = self.agent_states[current_idx].observation
            fen_string = obs.get("observationString")

            # 파싱 및 점수 계산
            board = stub.FENParser.parse(fen_string)
            board_score = stub.BoardEvaluator.get_material_score(board)

            # 출력
            self._display_board(board, board_score)

            print(f"--- 현재 턴: {player_name} (idx: {current_idx}) ---")
            print(self.env.render(mode="ansi"))

            legal_actions = obs.get("legalActions", [])
            legal_strings = obs.get("legalActionStrings", [])

            if not legal_actions:
                print("더 이상 둘 수 있는 수가 없습니다.")
                break
            
            # 액션 선택 및 수행
            chosen_action = stub.ActionSelector.choose(legal_actions, legal_strings)
            actions = [None, None]
            actions[current_idx] = {"submission": chosen_action}
            
            self.agent_states = self.env.step(actions)
            
            print(f"Player {current_idx} ({player_name})가 액션 '{chosen_action}'를 수행했습니다.")
            time.sleep(1)

        # 게임 종료 처리
        self._clear_screen()
        print(self.env.render(mode="ansi"))
        print("게임 종료!")
        self._save_result()

if __name__ == "__main__":
    game = ChessGameManager()
    game.play()